# Copyright 2020 Huawei Technologies Co., Ltd
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



def get_lr(lr_max, total_steps):
    """"get_learning_rate"""
    decay_epoch_index = [0.4* total_steps, 0.8 * total_steps, total_steps]
    lr_each_step = []
    for i in range(total_steps):

        if i < decay_epoch_index[0]:
            lr = lr_max
        elif i < decay_epoch_index[1]:
            lr = lr_max * 0.1
        elif i < decay_epoch_index[2]:
            lr = lr_max * 0.01
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step
