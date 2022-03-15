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
"""train"""
import os

import numpy as np
from mindspore import context
from mindspore import nn
from mindspore import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init

from model_utils.config import config
from src.callbacks import get_callbacks
from src.criterion import build_criterion
from src.dataset import build_dataset
from src.detr import TrainOneStepCellWithSense
from src.detr import TrainOneStepWrapper
from src.detr import build_detr
from src.utils import check_args


def set_default():
    """set default"""
    check_args(config)
    set_seed(config.seed)
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target)
    # init distributed
    if config.is_distributed:
        init('nccl')
        config.rank = get_rank()
        config.device_num = get_group_size()
        context.reset_auto_parallel_context()
        parallel_mode = context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=config.device_num)
    else:
        config.rank = 0
        config.device_num = 1
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    config.save_ckpt_logs = config.rank == 0


def get_optimizer(model):
    """get optimizer"""
    lr = nn.piecewise_constant_lr(
        [config.steps_per_epoch * config.lr_drop, config.steps_per_epoch * config.epochs],
        [config.lr, config.lr * 0.1]
    )
    lr_backbone = nn.piecewise_constant_lr(
        [config.steps_per_epoch * config.lr_drop, config.steps_per_epoch * config.epochs],
        [config.lr_backbone, config.lr_backbone * 0.1]
    )
    param_dicts = [
        {
            'params': [par for par in model.trainable_params() if 'backbone' not in par.name],
            'lr': lr,
            'weight_decay': config.weight_decay
        },
        {
            'params': [par for par in model.trainable_params() if 'backbone' in par.name],
            'lr': lr_backbone,
            'weight_decay': config.weight_decay
        }
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)
    return optimizer


def prepare_train():
    """prepare train"""
    detr = build_detr(config)
    criterion = build_criterion(config)

    dataset, length_dataset = build_dataset(config)
    steps_per_epoch = int(length_dataset / config.batch_size / config.device_num)
    config.steps_per_epoch = steps_per_epoch

    optimizer = get_optimizer(detr)
    return detr, optimizer, criterion, dataset


def run_train():
    """run training process"""
    set_default()

    detr, optimizer, criterion, dataset = prepare_train()
    data_loader = dataset.create_dict_iterator()
    detr.set_train()

    if config.aux_loss:
        sens_param = np.ones([config.dec_layers, config.batch_size, config.num_queries, 96])
    else:
        sens_param = np.ones([config.batch_size, config.num_queries, 96])
    step_cell = TrainOneStepCellWithSense(detr, optimizer, sens_param, config.clip_max_norm)
    train_wrapper = TrainOneStepWrapper(step_cell, criterion,
                                        config.aux_loss, config.dec_layers)

    if config.save_ckpt_logs:
        callbacks = get_callbacks(config, detr)

    for sample in data_loader:
        images = sample['image']
        masks = sample['mask']
        out = train_wrapper([images, masks], sample)
        loss_value, _, _ = out
        if config.save_ckpt_logs:
            callbacks(loss_value)


if __name__ == "__main__":
    run_train()
