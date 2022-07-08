# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""OCRNet training."""
import argparse
import ast
import os

import numpy as np
from mindspore import context, Model
from mindspore import dataset as de
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import SGD
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.callback import EvalCallback
from src.cityscapes import Cityscapes
from src.config import config_hrnetv2_w48 as config
from src.config import organize_configuration
from src.loss import CrossEntropy
from src.model_utils.moxing_adapter import moxing_wrapper
from src.seg_hrnet_ocr import get_seg_model

set_seed(1)
de.config.set_seed(1)


def eval_callback(network, cfg, device_id=0):
    """Create an object for inference while training."""
    dataset = Cityscapes(cfg.data_path,
                         num_samples=None,
                         num_classes=cfg.dataset.num_classes,
                         multi_scale=False,
                         flip=False,
                         ignore_label=cfg.dataset.ignore_label,
                         base_size=cfg.eval.base_size,
                         crop_size=cfg.eval.image_size,
                         downsample_rate=1,
                         scale_factor=16,
                         mean=cfg.dataset.mean,
                         std=cfg.dataset.std,
                         is_train=False)
    data_vl = de.GeneratorDataset(dataset, column_names=["image", "label"],
                                  shuffle=False,
                                  num_parallel_workers=cfg.workers)
    data_vl = data_vl.batch(1, drop_remainder=True)
    eval_cb = EvalCallback(network, data_vl, cfg.dataset.num_classes,
                           cfg.dataset.ignore_label, cfg.output_path, eval_interval=cfg.eval_interval,
                           device_id=device_id)
    return eval_cb


def get_exp_lr(base_lr, xs, power=4e-10):
    """Get learning rates for each step."""
    ys = []
    for x in xs:
        ys.append(base_lr / np.exp(power * x ** 2))
    return ys


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore OCRNet Training Configurations.")
    parser.add_argument("--data_url", type=str, default=None, help="Storage path of dataset in OBS.")
    parser.add_argument("--train_url", type=str, default=None, help="Storage path of training results in OBS.")
    parser.add_argument("--data_path", type=str, default=None, help="Storage path of dataset on machine.")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Target device [Ascend, GPU]")
    parser.add_argument("--output_path", type=str, default=None, help="Storage path of training results on machine.")
    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming in OBS.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming on machine.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run on ModelArts or offline machines.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--lr", type=float, default=0.0012,
                        help="Base learning rate.")
    parser.add_argument("--lr_power", type=float, default=4e-10,
                        help="Learning rate adjustment power.")
    parser.add_argument("--begin_epoch", type=int, default=0,
                        help="If it's a training resuming task, give it a beginning epoch.")
    parser.add_argument("--end_epoch", type=int, default=1000,
                        help="If you want to stop the task early, give it an ending epoch.")
    parser.add_argument("--batchsize", type=int, default=3,
                        help="batch size.")
    parser.add_argument("--eval_callback", type=ast.literal_eval, default=False,
                        help="To use inference while training or not.")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Epoch interval of evaluating result during training.")
    return parser.parse_args()


@moxing_wrapper(config)
def main():
    """Training process."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID")) if config.device_target == "Ascend" else get_rank()
        device_num = int(os.getenv("RANK_SIZE")) if config.device_target == "Ascend" else get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=device_num)
    else:
        device_id = 0
        device_num = 1

    # Create dataset
    # prepare dataset for train
    crop_size = (config.train.image_size[0], config.train.image_size[1])
    data_tr = Cityscapes(config.data_path,
                         num_samples=None,
                         num_classes=config.dataset.num_classes,
                         multi_scale=config.train.multi_scale,
                         flip=config.train.flip,
                         ignore_label=config.dataset.ignore_label,
                         base_size=config.train.base_size,
                         crop_size=crop_size,
                         downsample_rate=config.train.downsample_rate,
                         scale_factor=config.train.scale_factor,
                         mean=config.dataset.mean,
                         std=config.dataset.std,
                         is_train=True)
    # dataset.show()
    if device_num == 1:
        dataset = de.GeneratorDataset(data_tr, column_names=["image", "label"],
                                      num_parallel_workers=config.workers,
                                      shuffle=config.train.shuffle)
    else:
        dataset = de.GeneratorDataset(data_tr, column_names=["image", "label"],
                                      num_parallel_workers=config.workers,
                                      shuffle=config.train.shuffle,
                                      num_shards=device_num, shard_id=device_id)
    dataset = dataset.batch(config.batchsize, drop_remainder=True)

    # Create network
    net = get_seg_model(config)
    net.set_train(True)

    # Create loss
    loss = CrossEntropy(num_classes=config.dataset.num_classes, ignore_label=255)
    loss_scale_manager = FixedLossScaleManager(config.loss.loss_scale, False)
    # Learning rate adjustment.
    steps_per_epoch = dataset.get_dataset_size()
    total_steps = config.total_epoch * steps_per_epoch
    begin_step = config.begin_epoch * steps_per_epoch
    end_step = config.end_epoch * steps_per_epoch
    xs = np.linspace(0, total_steps, total_steps)
    lr = get_exp_lr(config.lr, xs, power=config.lr_power)
    lr = lr[begin_step: end_step]
    # Optimizer
    params = [{'params': net.trainable_params()}]
    opt = SGD(params,
              lr,
              config.train.opt_momentum,
              config.train.wd)

    # Create model
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O3",
                  keep_batchnorm_fp32=False)
    # Callbacks
    time_cb = TimeMonitor(data_size=steps_per_epoch)
    loss_cb = LossMonitor(per_print_times=steps_per_epoch)
    # Save-checkpoint callback
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * config.save_checkpoint_epochs,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix='{}'.format("seg_OCRNet-SGD"),
                              directory=config.output_path + "/card" + str(device_id),
                              config=ckpt_config)
    cb = [time_cb, loss_cb, ckpt_cb]
    # Self-defined callbacks
    if config.eval_callback and device_id == 0:
        eval_cb = eval_callback(net, config, device_id)
        cb.append(eval_cb)

    train_epoch = config.end_epoch - config.begin_epoch
    model.train(train_epoch, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    args = parse_args()
    organize_configuration(cfg=config, args=args)
    main()
