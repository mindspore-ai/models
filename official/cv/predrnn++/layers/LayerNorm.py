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

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

EPSILON = 0.00001

class LayerNorm(nn.Cell):
    def __init__(self, in_channel):
        """Initialize the Gradient Highway Unit.
        """
        super(LayerNorm, self).__init__()
        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.expand_dims = P.ExpandDims()
        self.s = Parameter(initializer('ones', (in_channel,)))
        self.b = Parameter(initializer('zeros', (in_channel,)))
        self.in_channel = in_channel

    def construct(self, x):
        m = self.mean(x, [1, 2, 3])
        v = self.mean(self.square(x - m), [1, 2, 3])
        a = x-m
        aa = a/F.sqrt(v+1e-5)
        aaa = aa*P.Reshape()(self.s, (1, self.in_channel, 1, 1))
        aaaa = aaa+P.Reshape()(self.b, (1, self.in_channel, 1, 1))
        return aaaa
