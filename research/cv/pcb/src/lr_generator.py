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


def get_lr(lr_init, lr_max, total_steps, warmup_steps, decay_steps):
    """get the learning rate of each step"""
    decay_step_index = list(range(decay_steps, total_steps, decay_steps))
    decay_step_index.append(total_steps) # pivot for convenience
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            for j in range(len(decay_step_index)):
                if i < decay_step_index[j]:
                    lr = lr_max * pow(config.decay_rate, j)
                    break
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = Tensor(lr_each_step)
    return lr_each_step
