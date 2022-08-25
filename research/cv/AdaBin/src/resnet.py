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
from mindspore.ops import operations as P

from src.binarylib import AdaBinConv2d, Maxout

class LambdaLayer(nn.Cell):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Cell):
    """
    ResNet basic cell definition.

    Args:
        None.
    Returns:
        Tensor, output tensor.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = AdaBinConv2d(in_planes, planes, kernel_size=3, stride=stride, pad_mode="pad", padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = Maxout(planes)

        self.conv2 = AdaBinConv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = Maxout(planes)

        self.pad = nn.SequentialCell()
        if stride != 1 or in_planes != planes:
            self.pad = nn.Pad(((0, 0), (planes // 4, planes // 4), (0, 0), (0, 0)))

    def construct(self, x):
        """ construct """

        out = self.bn1(self.conv1(x))
        if self.stride != 1 or self.in_planes != self.planes:
            x = x[:, :, ::2, ::2]
        out += self.pad(x)
        out = self.nonlinear1(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = self.nonlinear2(out)
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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.nonlinear1 = Maxout(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.ap = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.bn2 = nn.BatchNorm1d(64)

        self.linear = nn.Dense(64, num_classes)

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
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """construct"""

        out = self.nonlinear1(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.ap(out, (2, 3))
        out = self.flatten(out)
        out = self.bn2(out)
        out = self.linear(out)

        return out

def resnet20():
    """ resnet20 """
    return ResNet(BasicBlock, [3, 3, 3])
