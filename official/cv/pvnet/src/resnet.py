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
"""ResNet18"""
import math

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Parameter

from src.net_utils import load_pretrained


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""

    kernel_size = 3

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, pad_mode="pad",
                     padding=full_padding, dilation=dilation, has_bias=False)


class BasicBlock(nn.Cell):
    """BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """__init__"""
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """construct"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """ResNet"""

    def __init__(self, block, layers, num_classes=1000, fully_conv=False, remove_avg_pool_layer=False,
                 output_stride=32):
        """__init__"""
        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.remove_avg_pool_layer = remove_avg_pool_layer
        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), 'CONSTANT')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Dense(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, stride=1, pad_mode="same")

        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                weight_shape = (m.out_channels, m.in_channels, m.kernel_size[0], m.kernel_size[1])
                m.weight = Parameter(mindspore.Tensor(np.random.normal(0, math.sqrt(2. / n), weight_shape),
                                                      mindspore.float32))

            elif isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """_make_layer"""
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:

                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:

                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)

        x2s_padding = self.pad(x2s)
        x = self.maxpool(x2s_padding)

        x4s = self.layer1(x)
        x8s = self.layer2(x4s)
        x16s = self.layer3(x8s)
        x32s = self.layer4(x16s)
        x = x32s

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        xfc = self.fc(x)
        return x2s, x4s, x8s, x16s, x32s, xfc


def resnet18(pretrained_path=None, **kwargs):
    """resnet18"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained_path is not None:
        model = load_pretrained(model, pretrained_path)
    return model
