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


def get_lr(lr_init, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    warmup_epoch = 2
    hold_epoch = 140
    warmup_step = warmup_epoch * steps_per_epoch
    hold_step = hold_epoch * steps_per_epoch
    total_step = total_epochs * steps_per_epoch

    lr_each_step = []
    for i in range(total_step):
        if i < warmup_step:
            a = (i+1)/(warmup_step+1)
        elif i < warmup_step + hold_step:
            a = 1.0
        else:
            epoch = int(i / steps_per_epoch) + 1
            a = 0.981 ** (epoch - hold_epoch - warmup_epoch)
        lr = max(a * lr_init, 0.00001)
        lr_each_step.append(lr)

    learning_rate = np.array(lr_each_step).astype(np.float32)
    return learning_rate
