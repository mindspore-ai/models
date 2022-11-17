# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Inception Resnet V2 backbone."""

import mindspore.ops as ops
import mindspore.nn as nn

class BasicConv2d(nn.Cell):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, pad_mode='pad', padding=padding, has_bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(num_features=out_planes, eps=0.001, momentum=0.9, affine=True)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Cell):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.SequentialCell([
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        ])

        self.branch2 = nn.SequentialCell([
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        ])

        self.branch3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = ops.Concat(1)((x0, x1, x2, x3))
        return out


class Block35(nn.Cell):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.SequentialCell([
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        ])

        self.branch2 = nn.SequentialCell([
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        ])

        self.conv2d = nn.Conv2d(in_channels=128, out_channels=320, kernel_size=1, \
        stride=1, pad_mode='pad', has_bias=True)
        self.relu = nn.ReLU()

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = ops.Concat(1)((x0, x1, x2))
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Cell):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.SequentialCell([
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        ])

        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = ops.Concat(1)((x0, x1, x2))
        return out


class Block17(nn.Cell):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.SequentialCell([
            BasicConv2d(1088, 128, kernel_size=1, stride=1), \
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 0, 3, 3)), \
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 3, 0, 0))
        ])

        self.conv2d = nn.Conv2d(in_channels=384, out_channels=1088, kernel_size=1, \
        stride=1, pad_mode='pad', has_bias=True)
        self.relu = nn.ReLU()

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = ops.Concat(1)((x0, x1))
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Cell):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.SequentialCell([
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        ])

        self.branch1 = nn.SequentialCell([
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        ])

        self.branch2 = nn.SequentialCell([
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        ])

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = ops.Concat(1)((x0, x1, x2, x3))
        return out


class Block8(nn.Cell):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.SequentialCell([
            BasicConv2d(2080, 192, kernel_size=1, stride=1), \
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 0, 1, 1)), \
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 1, 0, 0))
        ])

        self.conv2d = nn.Conv2d(in_channels=448, out_channels=2080, kernel_size=1, \
        stride=1, pad_mode='pad', has_bias=True)
        if not self.noReLU:
            self.relu = nn.ReLU()

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = ops.Concat(1)((x0, x1))
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Cell):
    def __init__(self, num_classes=1000):
        super(InceptionResNetV2, self).__init__()
        # Special attributes
        # self.input_space = None
        # self.input_size = (299, 299, 3)
        # self.mean = None
        # self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.SequentialCell([
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        ])
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.SequentialCell([
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        ])
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.SequentialCell([
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        ])
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(kernel_size=8, stride=8, pad_mode='valid')
        self.last_linear = nn.Dense(in_channels=1536, out_channels=num_classes)

    def construct(self, input_data):
        # 0
        x = self.conv2d_1a(input_data)

        # 1
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        c2 = x

        # 2
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        c3 = x

        # 3
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        c4 = x

        # 4
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        c5 = x

        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return c2, c3, c4, c5

    # def logits(self, features):
        # x = self.avgpool_1a(features)
        # x = ops.Reshape()(x, (ops.Shape()(x)[0], -1,))
        # x = self.last_linear(x)
        # return x

    # def construct(self, input):
        # x = self.features(input)
        # x = self.logits(x)
        # return x
