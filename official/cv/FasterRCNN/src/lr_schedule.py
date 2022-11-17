# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""lr generator for fasterrcnn"""
import math


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def dynamic_lr(config, steps_per_epoch):
    """dynamic learning rate generator"""
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


def multistep_lr(config, dataset_size):
    learning_rate = float(config.base_lr)
    milestones_index = 0
    lr = []
    for epoch in range(config.epoch_size):
        if milestones_index < len(config.milestones):
            if epoch == config.milestones[milestones_index]:
                learning_rate = learning_rate * 0.1
                milestones_index += 1

        for _ in range(dataset_size):
            lr.append(learning_rate)

    return lr
