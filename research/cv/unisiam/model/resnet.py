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
import math

import mindspore as ms
from mindspore import Tensor

def _conv3x3(in_channel, out_channel, stride=1):
    return ms.nn.Conv2d(in_channel, out_channel, kernel_size=3,
                        stride=stride, padding=1, pad_mode='pad')

def _conv1x1(in_channel, out_channel, stride=1):
    return ms.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)

def _bn(channels):
    return ms.nn.BatchNorm2d(channels)

class BasicBlock(ms.nn.Cell):
    expansion = 1

    def __init__(self,
                 in_channels,
                 channels,
                 stride,
                 base_width=64,
                 down_sample=None):
        super().__init__()

        self.conv1 = _conv3x3(in_channels, channels, stride=stride)
        self.bn1 = _bn(channels)
        self.relu = ms.nn.ReLU()
        self.conv2 = _conv3x3(channels, channels, stride=1)
        self.bn2 = _bn(channels)
        self.down_sample = down_sample

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None: identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(ms.nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride=1,
                 base_width=64,
                 down_sample=None):
        super().__init__()

        width = int(channels * (base_width / 64.0))

        self.conv1 = _conv1x1(in_channels, width, stride=1)
        self.bn1 = _bn(width)
        self.conv2 = _conv3x3(width, width, stride=stride)
        self.bn2 = _bn(width)
        self.conv3 = _conv1x1(width, channels * self.expansion, stride=1)
        self.bn3 = _bn(channels * self.expansion)
        self.relu = ms.nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.down_sample is not None: identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNetV1(ms.nn.Cell):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 in_channels=3,
                 base_width=64):
        super().__init__()

        self.input_channels = 64
        self.base_with = base_width

        self.conv1 = ms.nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                                  stride=2, pad_mode='pad', padding=3)
        self.bn1 = _bn(self.input_channels)
        self.relu = ms.nn.ReLU()
        self.max_pool = ms.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        block_channels = [64, 128, 256, 512]
        block_strides = [1, 2, 2, 2]
        self.layer1 = self.make_layer(block, block_channels[0], layers[0], stride=block_strides[0])
        self.layer2 = self.make_layer(block, block_channels[1], layers[1], stride=block_strides[1])
        self.layer3 = self.make_layer(block, block_channels[2], layers[2], stride=block_strides[2])
        self.layer4 = self.make_layer(block, block_channels[3], layers[3], stride=block_strides[3])

        self.pool = ms.ops.AdaptiveAvgPool2D((1, 1))
        self.flatten = ms.ops.Flatten()
        self.num_features = 512 * block.expansion
        self.fc = ms.nn.Dense(self.num_features, num_classes)

        for _, cell in self.cells_and_names():
            if isinstance(cell, ms.nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (ms.nn.BatchNorm2d, ms.nn.GroupNorm)):
                cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, (ms.nn.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))


    def make_layer(self,
                   block,
                   channels,
                   block_nums,
                   stride=1):
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = ms.nn.SequentialCell([
                _conv1x1(self.input_channels, channels * block.expansion, stride=stride),
                ms.nn.BatchNorm2d(channels * block.expansion)
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                base_width=self.base_with
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    base_width=self.base_with
                )
            )

        return ms.nn.SequentialCell(layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        if self.fc is not None: x = self.fc(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self._forward_impl(x)
        return x


def resnet10():
    return ResNetV1(BasicBlock, [1, 1, 1, 1])


def resnet18():
    return ResNetV1(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNetV1(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNetV1(BottleneckBlock, [3, 4, 6, 3])
