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

"""Generate network."""

import math

from mindspore import nn
from mindspore import ops as P
import mindspore.common.initializer as weight_init


class ChannelAttention(nn.Cell):
    """
    ChannelAttention: Since each channel of the feature map is considered as a feature detector, it is meaningful
    for the channel to focus on the "what" of a given input image;In order to effectively calculate channel attention,
    the method of compressing the spatial dimension of input feature mapping is adopted.
    """
    def __init__(self, in_channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = P.ReduceMean(keep_dims=True)
        self.max_pool = P.ReduceMax(keep_dims=True)
        self.fc = nn.SequentialCell(nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 16, kernel_size=1,
                                              has_bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=in_channel // 16, out_channels=in_channel, kernel_size=1,
                                              has_bias=False))
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        avg_out = self.avg_pool(x, -1)
        avg_out = self.fc(avg_out)
        max_out = self.max_pool(x, -1)
        max_out = self.fc(max_out)
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Cell):
    """
    SpatialAttention: Different from the channel attention module, the spatial attention module focuses on the
    "where" of the information part as a supplement to the channel attention module.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, pad_mode='pad', has_bias=False)
        self.concat = P.Concat(axis=1)
        self.sigmod = nn.Sigmoid()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.max_pool = P.ReduceMax(keep_dims=True)

    def construct(self, x):
        avg_out = self.reduce_mean(x, 1)
        max_out = self.max_pool(x, 1)
        x = self.concat((avg_out, max_out))
        x = self.conv1(x)

        return self.sigmod(x)


def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution with padding.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, has_bias=False)


class Bottleneck(nn.Cell):
    """
    Residual structure.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU()

        self.ca = ChannelAttention(out_channels * 4)
        self.sa = SpatialAttention()

        self.dowmsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.dowmsample is not None:
            residual = self.dowmsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    Overall network architecture.
    """
    def __init__(self, block, layers, num_classes=11, phase="train"):
        self.in_channels = 64
        super(ResNet, self).__init__()
        self.phase = phase
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.Linear = nn.Dense(64 * block.expansion, num_classes)
        dropout_ratio = 0.5
        self.dropout = nn.Dropout(dropout_ratio)
        self.softmax = nn.Softmax()
        self.print = P.Print()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x, 3)
        x = x.view(32, 256)
        x = self.Linear(x)
        x = self.softmax(x)

        return x

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            i += 1
            layers.append(block(self.in_channels, out_channels))

        return nn.SequentialCell(*layers)

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Batchnorm2D in the net.
        :return:
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2. / n), mean=0),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.weight.set_data(weight_init.initializer(weight_init.One(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))


def resnet50_cbam(phase="train", **kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], phase=phase, **kwargs)
    return model
