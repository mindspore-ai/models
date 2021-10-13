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
"""lr"""

def get_lr(lr, total_epochs, steps_per_epoch, lr_step, gamma):
    """get_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    lr_step = [i * steps_per_epoch for i in lr_step]

    for i in range(total_steps):
        if i < lr_step[0]:
            lr_each_step.append(lr)
        elif i < lr_step[1]:
            lr_each_step.append(lr * gamma)
        else:
            lr_each_step.append(lr * gamma * gamma)

    return lr_each_step
