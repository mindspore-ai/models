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
"""Train script."""
import json

import numpy as np
from mindspore import Model
from mindspore import context
from mindspore import dataset as ds
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.dataset.vision import transforms as vision
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from cfg.config import config as default_config
from src.darknet import DarkNet, ResidualBlock
from src.dataset import JointDataset
from src.model import JDE
from src.model import YOLOv3

set_seed(1)


def lr_steps(cfg, steps_per_epoch):
    """
    Init lr steps.
    """
    learning_rate = warmup_lr(
        cfg.lr,
        steps_per_epoch,
        cfg.epochs,
    )

    return learning_rate


def warmup_lr(lr5, steps_per_epoch, max_epoch):
    """
    Set lr for training with warmup and freeze backbone.

    Args:
        lr5 (float): Initialized learning rate.
        steps_per_epoch (int): Num of steps per epoch on one device.
        max_epoch (int): Num of training epochs.

    Returns:
        lr_each_step (np.array): Lr for every step of training for model params.
    """
    base_lr = lr5
    warmup_steps = 1000
    total_steps = int(max_epoch * steps_per_epoch)
    milestone_1 = int(0.5 * max_epoch * steps_per_epoch)
    milestone_2 = int(0.75 * max_epoch * steps_per_epoch)

    lr_each_step = []

    for i in range(total_steps):
        if i < warmup_steps:
            lr5 = base_lr * ((i + 1) / warmup_steps) ** 4
        elif warmup_steps <= i < milestone_1:
            lr5 = base_lr
        elif milestone_1 <= i < milestone_2:
            lr5 = base_lr * 0.1
        elif milestone_2 <= i:
            lr5 = base_lr * 0.01

        lr_each_step.append(lr5)

    lr_each_step = np.array(lr_each_step, dtype=np.float32)

    return lr_each_step


def set_context(cfg):
    """
    Set process context.

    Args:
        cfg: Config parameters.

    Returns:
        dev_target (str): Device target platform.
        dev_num (int): Amount of devices participating in process.
        dev_id (int): Current process device id..
    """
    dev_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=dev_target)

    if dev_target == 'GPU':
        if cfg.is_distributed:
            init(backend_name='nccl')
            dev_num = get_group_size()
            dev_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=dev_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
        else:
            dev_num = 1
            dev_id = cfg.device_id
            context.set_context(device_id=dev_id)
    else:
        raise ValueError("Unsupported platform.")

    return dev_num, dev_id


def init_callbacks(cfg, batch_number, dev_id):
    """
    Initialize training callbacks.

    Args:
        cfg: Config parameters.
        batch_number: Number of batches into one epoch on one device.
        dev_id: Current process device id.

    Returns:
        cbs: Inited callbacks.
    """
    loss_cb = LossMonitor(per_print_times=100)
    time_cb = TimeMonitor(data_size=batch_number)

    if cfg.is_distributed and dev_id != cfg.device_start:
        cbs = [loss_cb, time_cb]
    else:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=batch_number,
            keep_checkpoint_max=cfg.keep_checkpoint_max,
        )

        ckpt_cb = ModelCheckpoint(
            prefix="JDE",
            directory=cfg.logs_dir,
            config=config_ck,
        )

        cbs = [loss_cb, time_cb, ckpt_cb]

    return cbs


if __name__ == "__main__":
    config = default_config
    device_target = config.device_target

    rank_size, rank_id = set_context(config)

    with open(config.data_cfg_url) as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']

    dataset = JointDataset(
        config.dataset_root,
        trainset_paths,
        k_max=config.k_max,
        augment=True,
        transforms=vision.ToTensor(),
        config=config,
    )

    dataloader = ds.GeneratorDataset(
        dataset,
        column_names=config.col_names_train,
        shuffle=True,
        num_parallel_workers=4,
        num_shards=rank_size,
        shard_id=rank_id,
        max_rowsize=12,
        python_multiprocessing=True,
    )

    dataloader = dataloader.batch(config.batch_size, True)

    batch_num = dataloader.get_dataset_size()

    # Initialize backbone
    darknet53 = DarkNet(
        ResidualBlock,
        config.backbone_layers,
        config.backbone_input_shape,
        config.backbone_shape,
        detect=True,
    )

    # Load weights into backbone
    if config.ckpt_url is not None:
        if config.ckpt_url.endswith(".ckpt"):
            param_dict = load_checkpoint(config.ckpt_url)
        else:
            raise ValueError(f"Unsupported checkpoint extension: {config.ckpt_url}.")

        load_param_into_net(darknet53, param_dict)
        print(f"Load pre-trained backbone from: {config.ckpt_url}")
    else:
        print("Start without pre-trained backbone.")

    # Initialize FPN with YOLOv3 head
    yolov3 = YOLOv3(
        backbone=darknet53,
        backbone_shape=config.backbone_shape,
        out_channel=config.out_channel,
    )

    # Initialize train model with loss cell
    net = JDE(yolov3, default_config, dataset.nid, config.embedding_dim)

    # Initiate lr for training
    lr = lr_steps(config, batch_num)

    params = net.trainable_params()

    # Set lr scheduler
    group_params = [
        {'params': params, 'lr': lr},
        {'order_params': params},
    ]

    opt = nn.SGD(
        params=group_params,
        learning_rate=lr,
        momentum=config.momentum,
        weight_decay=config.decay,
    )

    model = Model(net, optimizer=opt)

    callbacks = init_callbacks(config, batch_num, rank_id)

    model.train(epoch=config.epochs, train_dataset=dataloader, callbacks=callbacks, dataset_sink_mode=False)
    print("train success")
