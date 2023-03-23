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
"""train efficientnet."""
import os
import ast
import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.common import dtype as mstype
from mindspore.common import set_seed

from src.callback import TimeLossMonitor, EvalCallBack
from src.utils import get_linear_lr, params_initializer
from src.models.effnet import EfficientNet
from src.dataset import create_imagenet
from src.loss import CrossEntropySmooth
from src.config import efficientnet_b1_config_ascend as config
from src.config import organize_configuration
from src.model_utils.moxing_adapter import moxing_wrapper


set_seed(1)


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="image classification training")
    # Path parameter
    parser.add_argument("--data_url", type=str, default=None, help="Dataset path")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--train_url", type=str, default=None, help="Train output path")
    parser.add_argument("--train_path", type=str, default=None, help="Train output path")
    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="resume training with existed checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="resume training with existed checkpoint")
    parser.add_argument("--eval_data_url", type=str, default=None, help="Eval dataset path")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Eval dataset path")
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate frequency.")
    # Model parameter
    parser.add_argument("--model", type=str, default="efficientnet-b1")
    # Platform parameter
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False, help="Run mode")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=("Ascend", "GPU"), help="run platform")
    # Train parameter
    parser.add_argument("--begin_epoch", type=int, default=0, help="Begin epoch")
    parser.add_argument("--end_epoch", type=int, default=350, help="End epoch")
    parser.add_argument("--total_epoch", type=int, default=350, help="total epochs")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="rmsprop")
    parser.add_argument("--lr", type=float, default=0.15, help="base lr")
    parser.add_argument("--lr_scheme", type=str, default="linear")
    parser.add_argument("--lr_end", type=float, default=5e-5)
    args_opt = parser.parse_args()

    return args_opt


@moxing_wrapper(config)
def main():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = int(os.getenv("RANK_SIZE"))
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=device_num)
    else:
        device_id = 0
        device_num = 1
    print("Generating {}...".format(config.model), flush=True)
    dataset = create_imagenet(dataset_path=config.data_path, do_train=True, repeat_num=1,
                              input_size=config.input_size, batch_size=config.batchsize,
                              target=config.device_target, distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    if config.eval_data_path:
        eval_data = create_imagenet(dataset_path=config.eval_data_path, do_train=False, repeat_num=1,
                                    input_size=config.input_size, batch_size=config.batchsize,
                                    target=config.device_target, distribute=False)
    else:
        eval_data = None

    net = EfficientNet(width_coeff=config.width_coeff, depth_coeff=config.depth_coeff,
                       dropout_rate=config.dropout_rate, drop_connect_rate=config.drop_connect_rate,
                       num_classes=config.num_classes)
    if config.checkpoint_path:
        params = load_checkpoint(config.checkpoint_path)
        load_param_into_net(net, params)
    else:
        params_initializer(config, net)
    net.to_float(mstype.float16)
    net.set_train(True)

    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    # get learning rate
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = get_linear_lr(config.lr, config.total_epoch, step_size,
                       config.lr_init, config.lr_end,
                       warmup_epoch=config.warmup_epochs)
    lr = lr[config.begin_epoch * step_size: config.end_epoch * step_size]

    # define optimization
    optimizer = nn.RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9,
                           weight_decay=config.wd, momentum=config.opt_momentum,
                           epsilon=config.eps, loss_scale=config.loss_scale)

    # define model
    metrics = {
        "Top1-Acc": nn.Top1CategoricalAccuracy(),
        "Top5-Acc": nn.Top5CategoricalAccuracy()
    }
    model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                  metrics=metrics, amp_level="O3")

    # define callbacks
    cb = [TimeLossMonitor(lr_init=lr)]
    # Save-checkpoint callback
    if config.save_ckpt:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size*config.save_checkpoint_epochs,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=f"{config.model}-{config.dataset}",
                                  directory=os.path.join(config.train_path, f"card{device_id}"),
                                  config=ckpt_config)
        cb.append(ckpt_cb)
    if config.eval_data_path:
        eval_cb = EvalCallBack(model, eval_data, config.eval_interval, save_path=config.train_path)
        cb.append(eval_cb)
    # begine train
    epoch_size = config.end_epoch - config.begin_epoch
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == "__main__":
    args = parse_args()
    organize_configuration(config, args=args)
    main()
