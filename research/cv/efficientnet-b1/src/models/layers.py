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
"""efficientnet."""
import mindspore
import mindspore.numpy as np
from mindspore import Tensor, Parameter, nn
import mindspore.ops.operations as P


class Convolution(nn.Cell):
    """Convolution"""
    def __init__(self, in_, out_, kernel_size, stride=1, pad_mode="same", padding=0, groups=1, bias=False):
        super(Convolution, self).__init__()
        self.out = out_
        self.is_bias = bias
        self.bias = Parameter(Tensor(np.zeros((1, self.out, 1, 1)), mindspore.float32))
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride,
                              pad_mode=pad_mode, padding=padding, group=groups, has_bias=False)

    def construct(self, x):
        if self.is_bias:
            x = self.conv(x) + self.bias
        else:
            x = self.conv(x)
        return x


def conv_bn_act(in_, out_, kernel_size,
                stride=1, groups=1, bias=True,
                eps=1e-3, momentum=0.01):
    """conv_bn_act"""
    return nn.SequentialCell([
        Convolution(in_, out_, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm2d(num_features=out_, eps=eps, momentum=1.0 - momentum),
        Swish()
    ])


class Swish(nn.Cell):
    """Swish"""
    def construct(self, x):
        sigmoid = P.Sigmoid()
        x = x * sigmoid(x)
        return x


class Flatten(nn.Cell):
    """Flatten"""
    def construct(self, x):
        shape = P.Shape()
        reshape = P.Reshape()
        x = reshape(x, (shape(x)[0], -1))
        return x


class SEModule(nn.Cell):
    """SEModule"""
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.SequentialCell([
            AdaptiveAvgPool2d(),
            Convolution(in_, squeeze_ch, kernel_size=1, stride=1, pad_mode='pad', padding=0, bias=True),
            Swish(),
            Convolution(squeeze_ch, in_, kernel_size=1, stride=1, pad_mode='pad', padding=0, bias=True),
        ])

    def construct(self, x):
        sigmoid = P.Sigmoid()
        x = x * sigmoid(self.se(x))
        return x


class AdaptiveAvgPool2d(nn.Cell):
    """AdaptiveAvgPool2d"""
    def __init__(self):
        super(AdaptiveAvgPool2d, self).__init__()
        self.mean = P.ReduceMean(True)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x

class DropConnect(nn.Cell):
    """DropConnect"""
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def construct(self, x):
        """DropConnect"""
        if not self.training:
            return x

        random_tensor = self.ratio
        shape = (random_tensor.shape[0], 1, 1, 1)
        stdnormal = P.StandardNormal(seed=2)
        random_tensor = stdnormal(shape)
        random_tensor.requires_grad = False
        floor = P.Floor()
        x = x / self.ratio * floor(random_tensor)
        return x
