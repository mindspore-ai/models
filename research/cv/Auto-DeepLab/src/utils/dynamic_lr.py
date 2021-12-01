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
# ===========================================================================
"""Dynamic learning rate"""


class WarmUpPolyDecayLR:
    """Poly Learning Rate Scheduler with Warm Up"""
    def __init__(self, warmup_start_lr, base_lr, min_lr, warmup_iters, max_iteration):

        print('Using Poly LR Scheduler!')
        self.lr = base_lr
        self.max_iteration = max_iteration

        self.warmup_iters = warmup_iters
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_factor = (self.lr / warmup_start_lr) ** (1. / warmup_iters)

    def __call__(self, iteration):
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            lr = self.warmup_start_lr * (self.warmup_factor ** iteration)
        else:
            lr = self.lr * pow((1 - (iteration - self.warmup_iters) / (self.max_iteration - self.warmup_iters)), 0.9)
        return max(lr, self.min_lr)


def warmup_poly_lr(warmup_start_lr, base_lr, min_lr, warmup_iters, max_iteration):
    """List of warmup poly lr"""
    lr_scheduler = WarmUpPolyDecayLR(warmup_start_lr, base_lr, min_lr, warmup_iters, max_iteration)
    lr = []
    for i in range(max_iteration):
        lr.append(lr_scheduler(i))
    return lr
