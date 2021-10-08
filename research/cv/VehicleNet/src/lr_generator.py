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
"""lr generator"""
import math
import numpy as np

def lr_steps(lr_init=None, total_epochs=None, steps_per_epoch=None):
    """lr_steps"""
    learning_rate = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = total_steps * 2 / 3
    for i in range(total_steps):
        if i < decay_epoch_index:
            learning_rate.append(lr_init)
        else:
            learning_rate.append(lr_init * 0.1)
    learning_rate = np.array(learning_rate).astype(np.float32)

    return learning_rate

def lr_steps_1(lr_init=None, total_epochs=None, steps_per_epoch=None):
    """lr_steps_1"""
    learning_rate = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [total_steps * 3 / 4, total_steps * 15 / 16]        # 60, 75 / 80
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            learning_rate.append(lr_init)
        elif i < decay_epoch_index[1]:
            learning_rate.append(lr_init * 0.1)
        else:
            learning_rate.append(lr_init * 0.01)
    learning_rate = np.array(learning_rate).astype(np.float32)

    return learning_rate

def lr_steps_2(lr_init=None, total_epochs=None, steps_per_epoch=None):
    """lr_steps_2"""
    learning_rate = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = total_steps / 5            # 8 / 40
    for i in range(total_steps):
        if i < decay_epoch_index:
            learning_rate.append(lr_init)
        else:
            learning_rate.append(lr_init * 0.1)
    learning_rate = np.array(learning_rate).astype(np.float32)

    return learning_rate

def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    """get_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if lr_decay_mode == 'steps':
        decay_epoch_index = [total_steps * 3 / 4, total_steps * 15 / 16]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            else:
                lr = lr_max * 0.01
            lr_each_step.append(lr)
    elif lr_decay_mode == 'poly':
        if warmup_steps != 0:
            inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        else:
            inc_each_step = 0
        for i in range(total_steps):
            if i < warmup_steps:
                lr = float(lr_init) + inc_each_step * float(i)
            else:
                base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
                lr = float(lr_max) * base * base
                if lr < 0.0:
                    lr = 0.0
            lr_each_step.append(lr)
    elif lr_decay_mode == 'cosine':
        decay_steps = total_steps - warmup_steps
        for i in range(total_steps):
            if i < warmup_steps:
                lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
                lr = float(lr_init) + lr_inc * (i + 1)
            else:
                linear_decay = (total_steps - i) / decay_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
                decayed = linear_decay * cosine_decay + 0.00001
                lr = lr_max * decayed
            lr_each_step.append(lr)
    else:
        for i in range(total_steps):
            if i < warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else:
                lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step
