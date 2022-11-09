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
"""ResNet model define"""
import mindspore.nn as nn
import mindspore.ops as ops
from .norm import USNorm, GroupBatchNorm2d
from .base import _conv3x3, _conv1x1, _fc


BatchNorm2d = nn.BatchNorm2d
Conv2d = nn.Conv2d


class BasicBlock(nn.Cell):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = _conv3x3(in_planes, planes, stride=stride, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=1, bias=False)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                _conv1x1(in_planes, self.expansion*planes, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )
        self.relu = nn.ReLU()

    def construct(self, x):

        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    '''Bottleneck.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = _conv1x1(in_planes, planes, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=stride, bias=False)
        self.bn3 = BatchNorm2d(self.expansion*planes)
        self.conv3 = _conv1x1(planes, self.expansion*planes, bias=False)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                _conv1x1(in_planes, self.expansion*planes, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )
        self.relu = nn.ReLU()

    def construct(self, x):

        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10, normalize=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = _conv3x3(3, 64, stride=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = _fc(512 * block.expansion, num_classes)
        self.relu = nn.ReLU()
        self.avg_pool2d = ops.AvgPool(kernel_size=4, strides=4)
        self.normalize = normalize
        self.flatten = nn.Flatten()
        self.print = ops.Print()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for _stride in strides:
            layers.append(block(self.in_planes, planes, _stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)

    def construct(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


    def set_norms(self, norm=None):
        for module in self.cells_and_names():
            if isinstance(module, USNorm):
                module.set_norms(norm)

def ResNet18(norm_type, num_classes=10, normalize=None):
    global BatchNorm2d
    if isinstance(norm_type, list):
        BatchNorm2d = lambda num_features: USNorm(num_features, norm_type)
    elif isinstance(norm_type, str):
        if 'gn_' in norm_type:
            num_group = int(norm_type[norm_type.index('_')+1:])
            BatchNorm2d = lambda num_features: nn.GroupNorm(num_group, num_features)
        elif norm_type == 'bn':
            BatchNorm2d = nn.BatchNorm2d
        elif norm_type == 'in':
            BatchNorm2d = nn.InstanceNorm2d
        elif 'gbn_' in norm_type:
            num_group = int(norm_type[norm_type.index('_') + 1:])
            BatchNorm2d = lambda num_features: GroupBatchNorm2d(num_group, num_features)
        else:
            print('Wrong norm type.')
            exit()

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, normalize=normalize)
