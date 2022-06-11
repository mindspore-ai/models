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
"""lr generator for FasterRcnn-DCN"""

import math


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    """
    Scheduler for linear warmup of learning rate

    Args:
        current_step: Number of the current step
        warmup_steps: Number of steps to warmup
        base_lr: Base value of learning rate
        init_lr: Initial value of learning rate

    Returns:
        Current value of learning rate
    """
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    """
    Generate values for cosine annealing

    Args:
        current_step: Number of the current step
        base_lr: Base value of learning rate
        warmup_steps: Number of steps to warmup
        decay_steps: General number of steps

    Returns:
        Current value of learning rate
    """
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def dynamic_lr(config, steps_per_epoch):
    """
    Dynamic learning rate generator

    Args:
        config: Config object with training parameters
        steps_per_epoch: Number of steps per epoch

    Returns:
        List of learning rate values
    """
    base_lr = config.base_lr
    total_steps = steps_per_epoch * (config.epoch_size + 1)
    warmup_steps = int(config.warmup_step)
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * config.warmup_ratio))
        else:
            lr.append(a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))

    return lr
