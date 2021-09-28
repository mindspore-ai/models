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

import numpy as np


def linear_warmup_learning_rate(lr_max, epoch_step, global_step=0, lr_init=1e-8,
                                warmup_epochs=0, total_epochs=1, steps_per_epoch=1):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    lr_value = lr_max
    for i in range(total_steps):
        if i <= warmup_steps:
            lr_value = float(lr_init) + inc_each_step * float(i)
        else:
            if i // steps_per_epoch in epoch_step and i % steps_per_epoch == 0:
                lr_value *= 0.1
            if lr_value < 0.0:
                lr_value = 0.0
        lr_each_step.append(lr_value)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[global_step:]

    return learning_rate
