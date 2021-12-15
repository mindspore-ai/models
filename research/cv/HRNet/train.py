# Copyright 2021 Huawei Technologies Co., Ltd
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
"""HRNet training."""
import os
import ast
import argparse
import numpy as np

import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore import context, Model, save_checkpoint
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.seg_hrnet import get_seg_model
from src.loss import CrossEntropyWithWeights
from src.callback import TimeLossMonitor, SegEvalCallback
from src.config import task_config as config
from src.config import hrnetw48_config as model_config
from src.dataset.dataset_generator import create_seg_dataset


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """Remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key, flush=True)
                del origin_dict[key]
                break


def get_cos_lr(lr_max, lr_min, total_epoch, spe):
    """Get learning rates decaying in cosine annealing mode."""
    lr_min = lr_min
    lr_max = lr_max
    lrs = []
    total_step = spe * total_epoch
    for i in range(total_step):
        lrs.append(lr_min + (lr_max - lr_min) * (1 + np.cos(i * np.pi / total_step)) / 2)
    return lrs


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore HRNet Training Configurations.")
    parser.add_argument("--data_url", type=str, default=None, help="Storage path of dataset in OBS.")
    parser.add_argument("--dataset", type=str, default="cityscapes",
                        help="Dataset.")
    parser.add_argument("--train_url", type=str, default=None, help="Storage path of training results in OBS.")
    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint for pretraining or resuming in OBS.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run on ModelArts or offline machines.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--begin_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=600)
    parser.add_argument("--eval", type=ast.literal_eval, default=False)
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--interval", type=int, default=50)

    return parser.parse_args()


def main():
    """Training process."""
    set_seed(1)
    args = parse_args()

    if args.modelarts:
        import moxing as mox
        local_data_url = "/cache/dataset"
        mox.file.copy_parallel(args.data_url, local_data_url)
        local_train_url = "/cache/output"
        if args.checkpoint_url:
            if "obs://" in args.checkpoint_url:
                local_checkpoint_url = "/cache/" + args.checkpoint_url.split("/")[-1]
                mox.file.copy_parallel(args.checkpoint_url, local_checkpoint_url)
            else:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                ckpt_name = args.checkpoint_url[2:]
                local_checkpoint_url = os.path.join(dir_path, ckpt_name)
        else:
            local_checkpoint_url = None
    else:
        local_data_url = args.data_url
        local_train_url = args.train_url
        local_checkpoint_url = args.checkpoint_url

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if args.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = int(os.getenv("RANK_SIZE"))
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=device_num)
    else:
        device_id = 0

    # Create dataset
    loader, image_size, num_classes, class_weights = create_seg_dataset(
        args.dataset, local_data_url, config.batchsize, args.run_distribute, is_train=True)

    # Create network
    net = get_seg_model(model_config, num_classes)
    if local_checkpoint_url:
        pretrained_dict = load_checkpoint(local_checkpoint_url)
        if "hrnetw48seg" in local_checkpoint_url and args.dataset not in local_checkpoint_url:
            filter_list = [x.name for x in net.last_layer.get_parameters()]
            filter_checkpoint_parameter_by_list(pretrained_dict, filter_list)
        load_param_into_net(net, pretrained_dict)
    net.set_train(True)

    # Create loss
    loss = CrossEntropyWithWeights(num_classes=num_classes, ignore_label=255,
                                   image_size=image_size, weights=class_weights)
    loss_scale_manager = FixedLossScaleManager(config.loss_scale, False)
    # Learning rate adjustment.
    steps_per_epoch = loader.get_dataset_size()
    begin_step = args.begin_epoch * steps_per_epoch
    lr = get_cos_lr(config.lr, config.lr_min, config.total_epoch, steps_per_epoch)
    lr = lr[begin_step:]
    # Optimizer
    opt = nn.Adam(net.trainable_params(), learning_rate=lr, weight_decay=config.wd,
                  loss_scale=config.loss_scale, use_nesterov=True)

    # Create model
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O3",
                  keep_batchnorm_fp32=False)
    # Callbacks
    time_loss_cb = TimeLossMonitor(lr_init=lr)
    cb = [time_loss_cb]
    # Save-checkpoint callback
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * config.save_checkpoint_epochs,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=f"hrnetw48seg-{args.dataset}",
                              directory=os.path.join(local_train_url, f"card{device_id}"),
                              config=ckpt_config)
    cb.append(ckpt_cb)
    # Self-defined callbacks
    if args.eval:
        eval_loader, _, _, _ = create_seg_dataset(
            args.dataset, local_data_url, batchsize=1, run_distribute=False, is_train=False)
        eval_cb = SegEvalCallback(eval_loader, net, num_classes, start_epoch=args.eval_start,
                                  save_path=local_train_url, interval=args.interval)
        cb.append(eval_cb)

    if args.end_epoch > config.total_epoch:
        raise ValueError("End epoch should not be larger than total epoch.")
    train_epoch = args.end_epoch - args.begin_epoch
    model.train(train_epoch, loader, callbacks=cb, dataset_sink_mode=True)

    last_checkpoint = os.path.join(local_train_url, f"hrnetw48seg-{args.dataset}-{device_id}-final.ckpt")
    save_checkpoint(net, last_checkpoint)
    if args.modelarts:
        import moxing as mox
        mox.file.copy_parallel(local_train_url, args.train_url)


if __name__ == "__main__":
    main()
