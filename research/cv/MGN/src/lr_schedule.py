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
""" LR scheduler"""

import numpy as np


def get_exp_lr(lr_init, total_epochs, steps_per_epoch, start_decay_at_ep, gamma=0.001):
    """ Exponential learning rate scheduler """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs

    for i in range(total_steps):
        ep = np.floor(1. * i / steps_per_epoch) + 1
        if ep < start_decay_at_ep:
            lr = lr_init
        else:
            lr = (lr_init * (gamma ** (float(ep + 1 - start_decay_at_ep)
                                       / (total_epochs + 1 - start_decay_at_ep))))

        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    return lr_each_step


def get_step_lr(lr_init, total_epochs, steps_per_epoch, decay_epochs, gamma=0.1):
    """ Stepped learning rate scheduler """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    if isinstance(decay_epochs, str):
        decay_epochs = list(map(int, decay_epochs.split(',')))

    decay_epochs = decay_epochs.copy()
    mult = 1

    for i in range(total_steps):
        ep = np.floor(1. * i / steps_per_epoch) + 1
        if ep in decay_epochs:
            mult *= gamma
            decay_epochs.remove(ep)

        lr = lr_init * mult

        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    return lr_each_step
