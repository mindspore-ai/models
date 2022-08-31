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

from __future__ import division
from models.upsample import Upsample
import mindspore.nn as nn

class FCBlock(nn.Cell):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Dense(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.SequentialCell(*module_list)

    def construct(self, x):
        return self.fc_block(x)

class ConvBottleNeck(nn.Cell):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, nl_layer=nn.ReLU(), norm_type='GN'):
        super(ConvBottleNeck, self).__init__()

        self.nl_layer = nl_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, has_bias=True, bias_init='normal')
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, pad_mode="pad", padding=1,
                               has_bias=True, bias_init='normal')
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, has_bias=True, bias_init='normal')

        if norm_type == 'BN':
            affine = True
            self.norm1 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm2 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm3 = nn.BatchNorm2d(out_channels, affine=affine)
        else:
            self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=True, bias_init='normal')

    def construct(self, x):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nl_layer(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nl_layer(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.in_channels != self.out_channels:
            residual = self.skip_conv(residual)
        y += residual
        y = self.nl_layer(y)
        return y

# A net similar to hourglass, used in the UV net (Location Net)
class HgNet(nn.Cell):
    def __init__(self, in_channels, level, nl_layer=nn.ReLU(), norm_type='GN'):
        super(HgNet, self).__init__()

        down_layers = []
        up_layers = []

        if norm_type == 'GN':
            self.norm = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            affine = True
            self.norm = nn.BatchNorm2d(in_channels, affine=affine)

        for _ in range(level):
            out_channels = in_channels * 2
            down_layers.append(
                nn.SequentialCell(
                    ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nl_layer,
                                   norm_type=norm_type), nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                )
            up_layers.append(
                nn.SequentialCell(
                    Upsample(),
                    ConvBottleNeck(in_channels=out_channels, out_channels=in_channels,
                                   nl_layer=nl_layer, norm_type=norm_type)
                    )
                )
            in_channels = out_channels
        self.down_layers = nn.CellList(down_layers)
        self.up_layers = nn.CellList(up_layers)

    def construct(self, x):

        feature_list = []

        y = x
        for i in range(len(self.down_layers)):
            feature_list.append(y)
            y = self.down_layers[i](y)
        for i in range(len(self.down_layers) - 1, -1, -1):
            y = self.up_layers[i](y) + feature_list[i]
        y = self.norm(y)
        return y

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, has_bias=False)


def deconv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        Upsample(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, pad_model="valid")
    )
