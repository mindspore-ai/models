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
"""train_imagenet."""
import argparse
import ast
import os
import random

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config_gpu, config_ascend, config_cpu
from src.dataset import create_dataset_imagenet
from src.inception_v2 import inception_v2_base
from src.loss import CrossEntropy
from src.lr_generator import get_lr

CFG_DICT = {
    "Ascend": config_ascend,
    "GPU": config_gpu,
    "CPU": config_cpu,
}

DS_DICT = {
    "imagenet": create_dataset_imagenet
}


def set_random_seed(i):
    """sets random seed"""
    set_seed(i)
    np.random.seed(i)
    random.seed(i)


def run_train():
    """run train"""
    parser = argparse.ArgumentParser(description='image classification training')
    parser.add_argument("--data_url", type=str, help="dataset path.")
    parser.add_argument("--device_num", type=int, default=8, help="Use device nums, default is 8.")
    parser.add_argument("--train_url", type=str, help="train_out path.")
    parser.add_argument("--run_online", type=ast.literal_eval, default=False, help="whether run online.")
    parser.add_argument('--resume', type=str, default='', help='resume training with existed checkpoint')
    parser.add_argument("--is_distributed", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')
    args_opt = parser.parse_args()

    cfg = CFG_DICT[args_opt.platform]
    set_random_seed(cfg.random_seed)
    create_dataset = DS_DICT[cfg.ds_type]
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform, save_graphs=False)
    Imagenet_root = args_opt.data_url
    if not os.path.exists(args_opt.train_url):
        os.makedirs(args_opt.train_url, exist_ok=True)
    local_train_url = args_opt.train_url
    # create dataset on cache
    if args_opt.run_online:
        import moxing as mox

        Imagenet_root = "/cache/data_train"
        mox.file.copy_parallel(args_opt.data_url, Imagenet_root)
        local_train_url = "/cache/train_out_si"
    if args_opt.is_distributed:
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=cfg.group_size,
                                          gradients_mean=True)
    else:
        cfg.rank = 0
        cfg.group_size = 1
        if os.getenv('DEVICE_ID', "not_set").isdigit():
            context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # dataloader
    root = os.path.join(Imagenet_root, 'train')
    dataset = create_dataset(root, cfg, True)
    batches_per_epoch = dataset.get_dataset_size()
    net = inception_v2_base(num_classes=cfg.num_classes, dropout_keep_prob=cfg.dropout_keep_prob)

    # loss
    loss = CrossEntropy(smooth_factor=cfg.smooth_factor, num_classes=cfg.num_classes)

    # learning rate schedule
    lr = Tensor(get_lr(lr_init=cfg.lr_init, lr_end=cfg.lr_end, lr_max=cfg.lr_max, warmup_epochs=cfg.warmup_epochs,
                       total_epochs=cfg.epoch_size, steps_per_epoch=batches_per_epoch, lr_decay_mode=cfg.decay_method))
    group_params = filter(lambda x: x.requires_grad, net.get_parameters())
    opt = nn.Momentum(group_params, lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, loss_scale=cfg.loss_scale)

    if args_opt.resume != '':
        ckpt = load_checkpoint(args_opt.resume)
        load_param_into_net(net, ckpt)
    if args_opt.platform in ("Ascend", "GPU"):
        loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'}, amp_level=cfg.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'}, amp_level=cfg.amp_level)

    print("============== Starting Training ==============", flush=True)
    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=cfg.keep_checkpoint_max)
    save_ckpt_path = os.path.join(local_train_url, 'ckpt_' + str(cfg.rank) + '/')
    ckpoint_cb = ModelCheckpoint(prefix=f"inceptionv2-rank{cfg.rank}", directory=save_ckpt_path, config=config_ck)
    if args_opt.is_distributed and cfg.is_save_on_master:
        if cfg.rank == 0:
            callbacks.append(ckpoint_cb)
        model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=cfg.ds_sink_mode)
    else:
        callbacks.append(ckpoint_cb)
        model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=cfg.ds_sink_mode)
    if args_opt.run_online:
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
    print("train success", flush=True)


if __name__ == '__main__':
    run_train()
