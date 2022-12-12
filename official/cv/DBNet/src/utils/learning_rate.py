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
"""Warmup poly decay learning schedule."""
import numpy as np


def warmup_polydecay(base_lr, target_lr, warmup_epoch, total_epoch, start_epoch, steps_pre_epoch, factor=0.9):
    total_iters = total_epoch * steps_pre_epoch
    warmup_iters = warmup_epoch * steps_pre_epoch
    start_iters = start_epoch * steps_pre_epoch
    diff_lr = base_lr - target_lr
    lrs = []
    for it in range(start_iters, total_iters):
        if it <= warmup_iters:
            warmup_percent = min(it, warmup_iters) / warmup_iters
            lrs.append(base_lr * warmup_percent)
        else:
            cur_epoch = it // steps_pre_epoch - warmup_epoch
            poly_rate = (1.0 - cur_epoch / (total_epoch - warmup_epoch)) ** factor
            lrs.append(diff_lr * poly_rate + target_lr)
    return np.array(lrs, np.float32)
