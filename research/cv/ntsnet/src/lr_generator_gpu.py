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
"""lr generator for ntsnet"""
import math

import numpy as np


def warmup_cosine_annealing_lr(global_step, base_lr, steps_per_epoch, warmup_epochs, max_epoch, eta_min=1e-5):
    """ warmup cosine annealing lr."""
    warmup_init_lr = 0.0001
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(1, total_steps + 1):
        if i < warmup_steps:
            lr5 = warmup_init_lr + (base_lr - warmup_init_lr) * i / warmup_steps
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * i / total_steps)) / 2
        lr_each_step.append(lr5)
    lr_each_step = lr_each_step[global_step:]
    return np.array(lr_each_step).astype(np.float32)


def step_lr(global_step, lr_init, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_step):
    """
    generate learning rate
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        elif i < lr_step[0] * steps_per_epoch:
            lr = lr_max
        elif i < lr_step[1] * steps_per_epoch:
            lr = lr_max * 0.01
        else:
            lr = lr_max * 0.001
        lr_each_step.append(lr)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate
