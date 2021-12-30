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
"""
ResNet12
"""
from mindspore import nn
from mindspore.common.initializer import HeNormal, Constant
from mindspore.common import initializer as init


def weight_variable_conv():
    """
    :return: HeNormal
    """
    return HeNormal(mode='fan_out', nonlinearity='leaky_relu')


def weight_variable_bn(value):
    """
    :param value: constant value
    :return: Constant
    """
    return Constant(value)


def conv3x3(in_planes, out_planes):
    """
    :param in_planes: in_planes
    :param out_planes: out_planes
    :return: conv3x3
    """

    return nn.Conv2d(in_planes, out_planes, 3, padding=1, pad_mode='pad', has_bias=False)


def conv1x1(in_planes, out_planes):
    """
    :param in_planes: in_planes
    :param out_planes: out_planes
    :return: conv1x1
    """

    return nn.Conv2d(in_planes, out_planes, 1, has_bias=False)


def norm_layer(planes):
    """
    :param planes: planes
    :return: BatchNorm2d
    """

    return nn.BatchNorm2d(planes, momentum=0.1)


class Block(nn.Cell):
    """
    Block
    """

    def __init__(self, inplanes, planes, downsample):
        super(Block, self).__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample
        self.meanpool = nn.AvgPool2d(2, 2)

    def construct(self, x):
        """
        :param x: feat
        :return: block feat
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.meanpool(out)
        return out


class ResNet12(nn.Cell):
    """
    ResNet12
    """

    def __init__(self, channels):
        super(ResNet12, self).__init__()

        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.emb_size = channels[3]

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    HeNormal(negative_slope=0, mode='fan_out', nonlinearity='leaky_relu'),
                    cell.weight.shape, cell.weight.dtype))

    def _make_layer(self, planes):
        downsample = nn.SequentialCell(
            [conv1x1(self.inplanes, planes),
             norm_layer(planes)]
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def construct(self, x):
        """
        :param x: data
        :return: feat
        """
        x = self.layer1(x)  # 40*40
        x = self.layer2(x)  # 20*20
        x = self.layer3(x)  # 10*10
        x = self.layer4(x)  # 5*5
        x = x.view(x.shape[0], x.shape[1], -1).mean(axis=2)

        return x


def resnet12():
    """
    :return: resnet12
    """
    return ResNet12([64, 128, 256, 512])
