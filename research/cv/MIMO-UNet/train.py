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
"""
train MIMO_UNet
"""

import random
from pathlib import Path

import numpy as np
from mindspore import context
from mindspore import dataset as ds
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from src.config import config
from src.data_load import create_dataset_generator
from src.loss import ContentLoss
from src.metric import PSNR
from src.mimo_unet import MIMOUNet


def prepare_context(args):
    """prepare context"""
    context.set_context(mode=context.GRAPH_MODE)
    if args.is_train_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=args.group_size,
                                          gradients_mean=True)
    else:
        args.rank = 0
        args.group_size = 1
        context.set_context(device_id=args.device_id)


def prepare_dataset(args):
    """prepare dataset"""
    train_dataset_generator = create_dataset_generator(Path(args.dataset_root, 'train'),
                                                       make_aug=True)
    args.train_dataset_len = len(train_dataset_generator)
    train_dataset = ds.GeneratorDataset(train_dataset_generator, ["image", "label"],
                                        shuffle=True, num_parallel_workers=args.num_worker,
                                        num_shards=args.group_size, shard_id=args.rank)

    train_dataset = train_dataset.batch(batch_size=args.train_batch_size, drop_remainder=True)
    return train_dataset


def prepare_optimizer(net, args):
    """prepare optimizer"""
    lr = args.learning_rate
    lr_list = []
    for n_epoch in range(args.epochs_num):
        for _ in range(args.train_dataset_len // args.train_batch_size // args.group_size):
            lr_list.append(lr)
        if (n_epoch + 1) % 500 == 0:
            lr /= 2

    optim = nn.Adam(net.trainable_params(), beta1=0.9, beta2=0.999, learning_rate=lr_list)
    return optim


def prepare_callbacks(net, args):
    """prepare callbacks"""
    step_per_epoch = (args.train_dataset_len // args.group_size // args.train_batch_size)
    if args.rank == 0:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_save_frequency * step_per_epoch,
                                       saved_network=net,
                                       keep_checkpoint_max=40)
        ckpoint_cb = ModelCheckpoint(prefix='MIMO-UNet', directory=args.ckpt_save_directory,
                                     config=ckpt_config)
        train_callbacks = [
            LossMonitor(step_per_epoch),
            TimeMonitor(),
            ckpoint_cb,
        ]
    else:
        train_callbacks = [
            LossMonitor(step_per_epoch),
            TimeMonitor(),
        ]
    return train_callbacks


def train(args):
    """train"""

    random.seed(1)
    set_seed(1)
    np.random.seed(1)

    prepare_context(args)
    print(f"info rank {args.rank}, groupsize {args.group_size}")

    net = MIMOUNet()
    content_loss = ContentLoss()

    train_dataset = prepare_dataset(args)
    optim = prepare_optimizer(net, args)
    model = Model(net, content_loss, optim, metrics={"PSNR": PSNR()})

    train_callbacks = prepare_callbacks(net, args)
    print("train...")
    model.train(args.epochs_num, train_dataset,
                callbacks=train_callbacks,
                dataset_sink_mode=True)


if __name__ == '__main__':
    train(config)
