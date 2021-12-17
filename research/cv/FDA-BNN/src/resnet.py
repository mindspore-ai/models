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
"""ResNet."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from src.quan import QuanConv as MyConv

class LambdaLayer(nn.Cell):
    """ lambdalayer """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def construct(self, x):
        """ construct """
        return self.lambd(x)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel)


def _bn_last(channel):
    return nn.BatchNorm2d(channel)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class PReLU(nn.Cell):
    """ prelu cell """
    def __init__(self, channel=1, w=0.25):
        """Initialize PReLU."""
        super(PReLU, self).__init__()
        validator.check_positive_int(channel, 'channel', self.cls_name)
        if isinstance(w, (float, np.float32)):
            tmp = np.empty((channel,), dtype=np.float32)
            tmp.fill(w)
            w = Tensor(tmp, dtype=mstype.float32)
        elif isinstance(w, list):
            if len(w) != channel:
                raise ValueError("When the 'w' is a list, the length should be equal to the channel, "
                                 "but got the length  {len(w)}, the channel {channel}")
            for i in w:
                if not isinstance(i, (float, np.float32)):
                    raise ValueError("When the 'w' is a list, the all elements should be float, but got {w}")
            w = Tensor(w, dtype=mstype.float32)
        elif isinstance(w, Tensor):
            if w.dtype not in (mstype.float16, mstype.float32):
                raise ValueError("When the 'w' is a tensor, the dtype should be float16 or float32, but got {w.dtype}")
            if len(w.shape) != 1 or w.shape[0] != channel:
                raise ValueError("When the 'w' is a tensor, the rank should be 1, and the elements number "
                                 "should be equal to the channel, but got w shape {w}, the channel {channel}")
        else:
            raise TypeError("The 'w' only supported float list and tensor, but got {type(w)}")
        self.w = Parameter(w, name='a')
        self.prelu = P.PReLU()
        self.relu = P.ReLU()
        self.assign = P.Assign()

    def construct(self, x):
        """ construct """
        u = self.w
        v = self.prelu(x, F.cast(u, x.dtype))
        if self.training:
            self.assign(self.w, u)
        return v


class BasicCell(nn.Cell):
    """
    ResNet basic cell definition.

    Args:
        None.
    Returns:
        Tensor, output tensor.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicCell, self).__init__()

        self.conv1 = MyConv(in_planes, planes, 3, stride=stride, padding=1)
        self.bn1 = _bn(planes)
        self.conv2 = MyConv(planes, planes, 3, stride=1, padding=1)
        self.bn2 = _bn(planes)
        self.relu1 = PReLU()
        self.relu2 = PReLU()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        self.pad = nn.SequentialCell()
        if stride != 1 or in_planes != planes:
            self.pad = nn.Pad(((0, 0), (planes // 4, planes // 4), (0, 0), (0, 0)))

    def construct(self, x):
        """ construct """
        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.bn2(self.relu2(self.conv2(out)))
        if self.stride != 1 or self.in_planes != self.planes:
            x = x[:, :, ::2, ::2]
        x = self.pad(x)
        out += x
        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        cell (Cell): Cell for network.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = _conv3x3(3, 16)
        self.bn1 = _bn(16)

        self.ap = P.ReduceMean(keep_dims=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.flatten = nn.Flatten()
        self.linear = _fc(64, num_classes)
        self.relu = P.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        i = 0
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
            i += 2

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""

        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.ap(out, (2, 3))


        out = self.flatten(out)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    """ resnet20 """
    return ResNet(BasicCell, [3, 3, 3], num_classes)
