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

# This file was copied from project [openharmony][third_party_mindspore]

"""resnet56 with or without Non-Local block"""
import numpy as np
from scipy.stats import truncnorm

import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.common.initializer import HeNormal
import mindspore.common.initializer as weight_init

from src.models.non_local_embedded_gaussian import NONLocalBlock2D


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, pad_mode="pad", padding=1,
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=1,
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = None
        self.option = option
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = mindspore.ops.Pad(((0, 0), (planes // 4, planes // 4), (0, 0), (0, 0)))
            elif option == 'B':
                self.shortcut = nn.SequentialCell(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, has_bias=False,
                              weight_init=HeNormal(mode='fan_out')),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def construct(self, x):
        """construct"""
        ret = self.relu(self.bn1(self.conv1(x)))
        ret = self.bn2(self.conv2(ret))
        if self.shortcut is not None:
            if self.option == 'A':
                ret += self.shortcut(x[:, :, ::2, ::2])
            else:
                ret += self.shortcut(x)
        else:
            ret += x
        ret = self.relu(ret)
        return ret


class resnet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10, non_local=False):
        super(resnet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False,
                               weight_init=HeNormal(mode='fan_out'))
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        # add non-local block after layer 2
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, non_local=non_local)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Dense(64, num_classes)
        self.relu = mindspore.ops.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride, non_local=False):
        """make resnet layer"""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        last_idx = len(strides)
        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(block(self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion

        if non_local:
            layers.append(NONLocalBlock2D(in_channels=planes))
            layers.append(block(self.in_planes, planes, strides[-1]))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """construct"""
        ret = self.relu(self.bn1(self.conv1(x)))
        ret = self.layer1(ret)
        ret = self.layer2(ret)
        ret = self.layer3(ret)
        avgpool = mindspore.ops.AvgPool(ret.shape[3])
        ret = avgpool(ret)
        ret = ret.view(ret.shape[0], -1)
        ret = self.linear(ret)
        return ret


def resnet56(non_local=False, **kwargs):
    """Constructs a resnet-56 model"""
    return resnet(BasicBlock, [9, 9, 9], non_local=non_local, **kwargs)


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    """TruncatedNormal"""
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def init_weight(net, conv_init, dense_init):
    """init_weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            if conv_init == "XavierUniform":
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            elif conv_init == "TruncatedNormal":
                weight = conv_variance_scaling_initializer(cell.in_channels,
                                                           cell.out_channels,
                                                           cell.kernel_size[0])
                cell.weight.set_data(weight)
        if isinstance(cell, nn.Dense):
            if dense_init == "TruncatedNormal":
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            elif dense_init == "RandomNormal":
                in_channel = cell.in_channels
                out_channel = cell.out_channels
                weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
                cell.weight.set_data(weight)


def init_group_prams(network, weight_decay):
    """init_group_prams"""
    decayed_params = []
    no_decayed_params = []
    for param in network.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': network.trainable_params()}]
    return group_params
