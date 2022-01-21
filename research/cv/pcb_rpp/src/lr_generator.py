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
"""learning rate generator"""

import numpy as np
from mindspore import Tensor

from src.model_utils.config import config


def get_lr_decay(lr_init, lr_max, total_steps, warmup_steps):
    """ Get lr on each step with steps scheduler (on 0.3 and 0.7 steps) and with warmup """
    decay_epoch_index = [0.3 * total_steps, 0.7 * total_steps]
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * config.decay_rate
            else:
                lr = lr_max * config.decay_rate * config.decay_rate
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = Tensor(lr_each_step)
    return lr_each_step


def get_lr_constant(lr_init, lr_max, total_steps, warmup_steps):
    """ Get lr on each step with constant scheduler and with warmup """
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_max
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = Tensor(lr_each_step)
    return lr_each_step
