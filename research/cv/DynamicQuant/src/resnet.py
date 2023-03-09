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
"""
ResNet.
"""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from src.quant import DymQuanConv as MyConv
from src.quant import QuanConv
from src.gumbelsoftmax import GumbleSoftmax


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


def _conv3x3(in_planes, planes, stride=1):
    weight_shape = (planes, in_planes, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_planes, planes,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)


def _conv1x1(in_planes, planes, stride=1):
    weight_shape = (planes, in_planes, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_planes, planes,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_planes, planes, stride=1):
    weight_shape = (planes, in_planes, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_planes, planes,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel)


def _bn_last(channel):
    return nn.BatchNorm2d(channel)


def _fc(in_planes, planes):
    weight_shape = (planes, in_planes)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_planes, planes, has_bias=True, weight_init=weight, bias_init=0)


class BasicCell(nn.Cell):
    """
    ResNet basic cell definition.

    Args:
        None.
    Returns:
        Tensor, output tensor.
    """
    expansion = 1

    def __init__(self, in_planes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(BasicCell, self).__init__()

        self.conv1 = MyConv(in_planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=stride, padding=1)
        self.bn1 = _bn(planes)
        self.relu = nn.ReLU()
        self.conv2 = MyConv(planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=1, padding=1)
        self.bn2 = _bn(planes)

        self.downsample = downsample
        self.stride = stride

    def construct(self, x, one_hot):
        """ construct """
        residual = x

        out = self.conv1(x, one_hot)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, one_hot)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample[0](x, one_hot)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)
        return out


class QuanBasicCell(nn.Cell):
    """
    ResNet basic cell definition.

    Args:
        None.
    Returns:
        Tensor, output tensor.
    """
    expansion = 1

    def __init__(self, in_planes, planes, name_w, name_a, nbit_w, nbit_a, stride=1, downsample=None):
        super(QuanBasicCell, self).__init__()

        self.conv1 = QuanConv(in_planes, planes, 3, name_w, name_a, nbit_w, nbit_a,
                              stride=stride, padding=1, bias=False)
        self.bn1 = _bn(planes)
        self.relu = nn.ReLU()
        self.conv2 = QuanConv(planes, planes, 3, name_w, name_a, nbit_w, nbit_a, stride=1, padding=1, bias=False)
        self.bn2 = _bn(planes)

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """ construct """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)
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
    def __init__(self, block, num_blocks, name_w, name_a, nbit_w, nbit_a, num_bits=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = nn.ReLU()
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.layer1 = self._make_layer(QuanBasicCell, 64, num_blocks[0], name_w, name_a, nbit_w, nbit_a, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], name_w, name_a, nbit_w, nbit_a, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[2], name_w, name_a, nbit_w, nbit_a, stride=2)

        self.ap = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(p=0.2)
        self.fc = _fc(512 * block.expansion, num_classes)

        self.avgpool_policy = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc1 = _fc(64*8*8, 64)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc2 = _fc(64, num_bits)
        self.gumbelsoftmax = GumbleSoftmax()

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)
        x = self.maxpool(x)

        for m in self.layer1:
            x = m(x)

        middle = x

        feat = self.avgpool_policy(x)
        feat = self.fc1(feat.view(x.shape[0], -1))
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        one_hot = self.gumbelsoftmax(feat)

        for m in self.layer2:
            x = m(x, one_hot)
        for m in self.layer3:
            x = m(x, one_hot)
        for m in self.layer4:
            x = m(x, one_hot)

        x = self.ap(x, (2, 3))
        x = self.flatten(x)
        x = x.view(x.shape[0], -1)
        x = self.drop1(x)
        x = self.fc(x)
        return x, middle

    def _make_layer(self, block, planes, num_blocks, name_w, name_a, nbit_w, nbit_a, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_planes (int): Input channel.
            planes (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = []
            downsample.append(MyConv(self.in_planes, planes * block.expansion, 1, name_w,
                                     name_a, nbit_w, nbit_a, stride=stride, bias=False))
            downsample.append(_bn(planes * block.expansion))
            downsample = nn.CellList(downsample)

        layers = []
        layers.append(block(self.in_planes, planes, name_w, name_a, nbit_w, nbit_a, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, name_w, name_a, nbit_w, nbit_a))

        return nn.CellList(layers)


def resnet18(name_w='dorefa', name_a='dorefa', nbit_w=4, nbit_a=4, num_bits=3, num_classes=1000):
    """ resnet18 """
    return ResNet(BasicCell, [2, 2, 2, 2], name_w, name_a, nbit_w, nbit_a, num_bits, num_classes)
