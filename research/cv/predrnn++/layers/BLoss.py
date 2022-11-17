# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mindspore
from mindspore import nn
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

class BMSELoss(nn.Cell):
    def __init__(self):
        """Initialize the Gradient Highway Unit.
        """
        super(BMSELoss, self).__init__()
        self.square = P.Square()
        self.mean = P.ReduceMean()
        self.mul = P.Mul()
        self.ones = P.Ones()
        self.greater_equal = P.GreaterEqual()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()

    def construct(self, x, y):
        one = self.ones(y.shape, mindspore.float32)
        stage1 = self.greater_equal(y, 0.283)
        stage1 = self.cast(stage1, mstype.float32)
        stage2 = self.greater_equal(y, 0.353)
        stage2 = self.cast(stage2, mstype.float32)*3
        stage3 = self.greater_equal(y, 0.424)
        stage3 = self.cast(stage3, mstype.float32)*5
        stage4 = self.greater_equal(y, 0.565)
        stage4 = self.cast(stage4, mstype.float32)*20
        w = one + stage1 + stage2 + stage3 + stage4
        dist = self.square(y-x)
        aaa = self.reduce_sum(w*dist)
        return aaa/2
