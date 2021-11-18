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
"""adam optimizer with gradient clipping"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import functional as F

class AdamClipped(nn.Adam):
    """adam optimizer with gradient clipping"""
    def __init__(self, params, learning_rate, beta1, beta2, weight_decay, max_norm=10.0, norm_type=2.0):
        super().__init__(
            params, learning_rate=learning_rate, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.norm = nn.Norm()
        self.mul = ops.Mul()

    def construct(self, gradients):
        total_norm = 0.0
        for grad in gradients:
            total_norm += self.norm(grad) ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)
        clip_coef = self.max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            gradients = self.map_(F.partial(self.mul, clip_coef), gradients)
        return super().construct(gradients)
