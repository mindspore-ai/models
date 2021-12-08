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
"""DDRNet23 define"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common import initializer as weight_init

from .var_init import KaimingNormal


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath2D(DropPath):
    """DropPath2D"""

    def __init__(self, drop_prob):
        super(DropPath2D, self).__init__(drop_prob=drop_prob, ndim=2)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)


class BasicBlock(nn.Cell):
    """BasicBlock for DDRNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, drop_path_rate=0.):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu
        self.drop_path = DropPath2D(drop_prob=drop_path_rate)

    def construct(self, x):
        """BasicBlock construct"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += self.drop_path(residual)
        if self.no_relu:
            return out
        return self.relu(out)


class Bottleneck(nn.Cell):
    """Bottleneck for DDRNet"""
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, drop_path_rate=0.):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu
        self.drop_path = DropPath2D(drop_prob=drop_path_rate)

    def construct(self, x):
        """BasicBlock construct"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += self.drop_path(residual)
        if self.no_relu:
            return out
        return self.relu(out)


class DualResNet(nn.Cell):
    """DualResNet"""
    def __init__(self, block, layers, num_classes=1000, planes=64, last_planes=2048, image_size=224, drop_path_rate=0.):
        super(DualResNet, self).__init__()
        highres_planes = planes * 2
        self.drop_path_rate = np.linspace(0, drop_path_rate, sum(layers))
        self.last_planes = last_planes
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(3, planes, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes), nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU()])
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, planes, planes, layers[0],
                                       drop_path_rate=self.drop_path_rate[:sum(layers[:1])])
        self.layer2 = self._make_layer(
            block, planes, planes * 2, layers[1], stride=2,
            drop_path_rate=self.drop_path_rate[sum(layers[:1]):sum(layers[:2])])
        self.layer3 = self._make_layer(
            block, planes * 2, planes * 4, layers[2], stride=2,
            drop_path_rate=self.drop_path_rate[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layer(
            block, planes * 4, planes * 8, layers[3], stride=2,
            drop_path_rate=self.drop_path_rate[sum(layers[:3]):sum(layers[:4])])

        self.compression3 = nn.SequentialCell([
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1),
            nn.BatchNorm2d(highres_planes)])

        self.compression4 = nn.SequentialCell([
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1),
            nn.BatchNorm2d(highres_planes)])

        self.down3 = nn.SequentialCell([
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes * 4)])

        self.down4 = nn.SequentialCell([
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes * 8)])
        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2,
                                        drop_path_rate=self.drop_path_rate[sum(layers[:1]):sum(layers[:2])])

        self.layer4_ = self._make_layer(
            block, highres_planes, highres_planes, 2,
            drop_path_rate=self.drop_path_rate[sum(layers[:2]):sum(layers[:3])])

        self.layer5_ = self._make_layer(
            Bottleneck, highres_planes, highres_planes, 1, drop_path_rate=[drop_path_rate,])

        self.down5 = nn.SequentialCell([
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes * 8),
            nn.ReLU(),
            nn.Conv2d(planes * 8, planes * 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(planes * 16)])

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, drop_path_rate=[drop_path_rate,])

        self.last_layer = nn.SequentialCell([
            nn.Conv2d(planes * 16, last_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(last_planes),
            nn.ReLU()])

        self.pool = ops.ReduceMean(False)

        self.linear = nn.Dense(last_planes, num_classes)
        self.width_output = self.height_output = image_size // 8
        self.resize_1 = ops.ResizeBilinear(size=(self.width_output, self.height_output))
        self.resize_2 = ops.ResizeBilinear(size=(self.width_output, self.height_output))
        self.weigit_init()

    def weigit_init(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(KaimingNormal(mode='fan_out', nonlinearity='relu'),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    def _make_layer(self, block, inplanes, planes, blocks, drop_path_rate, stride=1):
        """make layer for ddrnet"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)])

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, drop_path_rate=drop_path_rate[0]))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True, drop_path_rate=drop_path_rate[i]))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, drop_path_rate=drop_path_rate[i]))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct for ddrnet"""
        x = self.conv1(x)
        x = self.layer1(x)
        layer2_out = self.layer2(self.relu(x))
        layer3_out = self.layer3(self.relu(layer2_out))
        x_ = self.layer3_(self.relu(layer2_out))

        x = layer3_out + self.down3(self.relu(x_))
        x_ = x_ + self.resize_1(self.compression3(self.relu(layer3_out)))
        layer4_out = self.layer4(self.relu(x))
        x_ = self.layer4_(self.relu(x_))

        x = layer4_out + self.down4(self.relu(x_))
        x_ = x_ + self.resize_2(self.compression4(self.relu(layer4_out)))
        x = self.layer5(self.relu(x)) + self.down5(self.relu(self.layer5_(self.relu(x_))))
        x = self.last_layer(self.relu(x))
        x = self.pool(x, [2, 3])
        x = self.linear(x)
        return x


def DDRNet23(args):
    """DDRNet23"""
    return DualResNet(block=BasicBlock, layers=[2, 2, 2, 2], drop_path_rate=args.drop_path_rate,
                      image_size=args.image_size, planes=64, last_planes=2048, num_classes=args.num_classes)
