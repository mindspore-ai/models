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
"""get learning rate"""
import numpy as np


def get_lr(init_lr, total_epoch, step_per_epoch, anneal_step=250):
    """warmup lr schedule"""
    total_step = total_epoch * step_per_epoch
    lr_step = []

    for step in range(total_step):
        lambda_lr = anneal_step ** 0.5 * \
                    min((step + 1) * anneal_step ** -1.5, (step + 1) ** -0.5)
        lr_step.append(init_lr * lambda_lr)
    learning_rate = np.array(lr_step).astype(np.float32)

    return learning_rate
