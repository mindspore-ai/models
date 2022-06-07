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

import mindspore.nn as nn
import mindspore.ops as ops
import cfg


class TokenExchange(nn.Cell):
    def __init__(self):
        super(TokenExchange, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.select = ops.Select()

    def construct(self, x, mask, mask_threshold):
        mask0 = self.expand_dims(mask[0].copy(), 2).expand_as(x[0])
        mask1 = self.expand_dims(mask[1].copy(), 2).expand_as(x[1])
        x0 = self.select(mask0 >= mask_threshold, x[0], x[1])
        x1 = self.select(mask1 < mask_threshold, x[0], x[1])
        return [x0, x1]


class ModuleParallel(nn.Cell):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def construct(self, x_parallel):
        return [self.module(x_parallel[0]), self.module(x_parallel[1])]


class LayerNormParallel(nn.Cell):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(cfg.num_parallel):
            setattr(self, "ln_" + str(i), nn.LayerNorm([num_features], epsilon=1e-6))

    def construct(self, x_parallel):
        return [
            getattr(self, "ln_0")(x_parallel[0]),
            getattr(self, "ln_1")(x_parallel[1]),
        ]
