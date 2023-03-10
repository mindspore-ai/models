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
"""WideResNet model define"""
import mindspore.nn as nn
import mindspore.ops as ops
from .norm import USNorm, GroupBatchNorm2d
from .base import _conv3x3, _conv1x1, _fc


BatchNorm2d = nn.BatchNorm2d


class BasicBlock(nn.Cell):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = _conv3x3(in_planes, out_planes, stride=stride, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = _conv3x3(out_planes, out_planes, stride=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.conv_shortcut = (_conv1x1(in_planes, out_planes, stride=stride, bias=False)
                             if not self.equalInOut else None)
        self.dropout = nn.Dropout(p=self.droprate)
        self.add = ops.Add()

    def construct(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = self.dropout(out)
        out = self.conv2(out)
        return self.add(x if self.equalInOut else self.conv_shortcut(x), out)


class NetworkBlock(nn.Cell):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.SequentialCell(layers)

    def construct(self, x):
        return self.layer(x)


class WideResNet(nn.Cell):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = _conv3x3(3, nChannels[0], stride=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.avg_pool2d = ops.AvgPool(kernel_size=8, strides=8)
        self.fc = _fc(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.flatten = nn.Flatten()

        self.normalize = normalize

    def construct(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        return self.fc(out)

    def set_norms(self, norm=None):
        for module in self.cells_and_names():
            if isinstance(module, USNorm):
                module.set_norms(norm)


def WideResNet32(norm_type, num_classes=10, normalize=None):
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

    return WideResNet(num_classes=num_classes, normalize=normalize)
