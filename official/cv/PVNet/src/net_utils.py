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
"""network utils"""
import numpy as np
from easydict import EasyDict
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint


def adjust_learning_rate(global_step, lr_init, lr_decay_rate, lr_decay_epoch, total_epochs, steps_per_epoch,
                         min_lr=1e-5):
    """adjust_learning_rate"""
    lr_per_epoch = [lr_init] * total_epochs * steps_per_epoch
    lr = lr_init
    for step in range(total_epochs * steps_per_epoch):
        if (step + 1) % (lr_decay_epoch * steps_per_epoch) == 0:
            lr = max(lr * lr_decay_rate, min_lr)
        lr_per_epoch[step] = lr

    lr_each_step = np.array(lr_per_epoch).astype(np.float32)
    return lr_each_step[global_step:]


class AverageMeter(EasyDict):
    """Computes and stores the average and current value"""

    def __init__(self):
        """__init__"""
        super().__init__()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_pretrained(net, ckpt_path):
    """load_pretrained"""
    param_dict = load_checkpoint(ckpt_path)
    for name, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            weight = '{}.weight'.format(name)
            bias = '{}.bias'.format(name)
            if weight in param_dict:
                cell.weight = param_dict[weight].data
            if bias in param_dict:
                cell.bias = param_dict[bias].data
        elif isinstance(cell, nn.BatchNorm2d):
            moving_mean = '{}.moving_mean'.format(name)
            moving_variance = '{}.moving_variance'.format(name)
            gamma = '{}.gamma'.format(name)
            beta = '{}.beta'.format(name)

            if moving_mean in param_dict:
                cell.moving_mean = param_dict[moving_mean].data
            if moving_variance in param_dict:
                cell.moving_variance = param_dict[moving_variance].data
            if gamma in param_dict:
                cell.gamma = param_dict[gamma].data
            if beta in param_dict:
                cell.beta = param_dict[beta].data
    print('load {} successfully'.format(ckpt_path))
    return net
