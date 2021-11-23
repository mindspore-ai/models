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

"""Inceptionv2"""

import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
import mindspore.ops as op
from mindspore.ops import operations as P


def weight_variable(stddev):
    """Weight variable."""
    return TruncatedNormal(stddev)


class Conv2dBlock(nn.Cell):
    """Define Conv2dBlock."""
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, pad_mode="same", is_variable=True):
        super(Conv2dBlock, self).__init__()
        self.is_variable = is_variable
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, weight_init="XavierUniform")
        self.BatchNorm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU6()

    def construct(self, x):
        x = self.conv(x)
        x = self.BatchNorm(x)
        x = self.relu(x)
        return x

class depthwise_separable_conv(nn.Cell):
    "Depthwise conv + Pointwise conv"
    def __init__(self, in_channels, out_channels, depthwise_multiplier,
                 kernel_size=1, stride=1, padding=0, pad_mode="same", stddev=1.0):
        super(depthwise_separable_conv, self).__init__()
        temp_channel = in_channels * depthwise_multiplier
        self.is_use_pointwise = out_channels
        if out_channels is not None:
            self.conv1 = nn.Conv2d(in_channels, temp_channel, kernel_size,
                                   stride, padding=padding, pad_mode=pad_mode,
                                   group=in_channels, weight_init=weight_variable(stddev))
            self.conv2 = nn.Conv2d(temp_channel, out_channels, kernel_size=1, stride=1, padding=0,
                                   weight_init=weight_variable(stddev))
        else:
            self.conv1 = nn.Conv2d(in_channels, temp_channel, kernel_size, stride, group=in_channels,
                                   padding=padding, pad_mode=pad_mode,
                                   weight_init=weight_variable(stddev))
        self.BatchNorm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.Relu = nn.ReLU6()

    def construct(self, x):
        if self.is_use_pointwise is not None:
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
        x = self.BatchNorm(x)
        x = self.Relu(x)
        return x


class Inception(nn.Cell):
    """
    Inception Block
    """
    def __init__(self, in_channels, n1x1, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2, pool_planes):
        super(Inception, self).__init__()
        self.Branch_0 = Conv2dBlock(in_channels, n1x1, kernel_size=1, is_variable=False)
        self.Branch_1 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_a, kernel_size=1),
                                           Conv2dBlock(n3x3red_a, n3x3, kernel_size=3, padding=0, is_variable=False)])
        self.Branch_2 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_b, kernel_size=1),
                                           Conv2dBlock(n3x3red_b, n3x3red_b_2, kernel_size=3,
                                                       padding=0, is_variable=False),
                                           Conv2dBlock(n3x3red_b_2, n3x3red_b_2, kernel_size=3, padding=0,
                                                       is_variable=False)])
        self.avgpool_op = op.AvgPool(pad_mode="SAME", kernel_size=3, strides=1)
        self.Branch_3 = Conv2dBlock(in_channels, pool_planes, kernel_size=1)
        self.concat = op.Concat(axis=1)

    def construct(self, x):
        branch1 = self.Branch_0(x)
        branch2 = self.Branch_1(x)
        branch3 = self.Branch_2(x)
        cell = self.avgpool_op(x)
        branch4 = self.Branch_3(cell)
        return self.concat((branch1, branch2, branch3, branch4))


class Inception_2(nn.Cell):
    """
    Inception_2 Block
    """
    def __init__(self, in_channels, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2):
        super(Inception_2, self).__init__()
        self.Branch_0 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_a, kernel_size=1),
                                           Conv2dBlock(n3x3red_a, n3x3, kernel_size=3, stride=2,
                                                       padding=0, is_variable=False)])

        self.Branch_1 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_b, kernel_size=1),
                                           Conv2dBlock(n3x3red_b, n3x3red_b_2, kernel_size=3,
                                                       padding=0, is_variable=False),
                                           Conv2dBlock(n3x3red_b_2, n3x3red_b_2, kernel_size=3,
                                                       padding=0, stride=2, is_variable=False)])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="SAME")
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.Branch_0(x)
        branch2 = self.Branch_1(x)
        branch3 = self.maxpool(x)
        return self.concat((branch1, branch2, branch3))


class inception_v2_base(nn.Cell):
    """details in inceptionv2."""
    def __init__(self, input_channels=3, use_separable_conv=False):
        super(inception_v2_base, self).__init__()
        self.feature_map_channels = {'Conv2d_1a_7x7': 64, 'MaxPool_2a_3x3': 64,
                                     'Conv2d_2b_1x1': 64, 'Conv2d_2c_3x3': 192,
                                     'MaxPool_3a_3x3': 192, 'Mixed_3b': 256,
                                     'Mixed_3c': 320, 'Mixed_4a': 576, 'Mixed_4b': 576,
                                     'Mixed_4c': 576, 'Mixed_4d': 576, 'Mixed_4e': 576,
                                     'Mixed_5a': 1024, 'Mixed_5b': 1024, 'Mixed_5c': 1024}
        if use_separable_conv:
            depthwise_multiplier = min(int(64 / 3), 8)
            self.Conv2d_1a_7x7 = depthwise_separable_conv(input_channels, 64,
                                                          depthwise_multiplier=depthwise_multiplier,
                                                          kernel_size=7, stride=2, padding=0)
        else:
            self.Conv2d_1a_7x7 = Conv2dBlock(input_channels, 64, kernel_size=7, stride=2)
        self.MaxPool_2a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.Conv2d_2b_1x1 = Conv2dBlock(64, 64, kernel_size=1)
        self.Conv2d_2c_3x3 = Conv2dBlock(64, 192, kernel_size=3, padding=0)
        self.MaxPool_3a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.Mixed_3b = Inception(192, 64, 64, 64, 64, 96, 32)
        self.Mixed_3c = Inception(256, 64, 64, 96, 64, 96, 64)
        self.Mixed_4a = Inception_2(320, 128, 160, 64, 96)
        self.Mixed_4b = Inception(576, 224, 64, 96, 96, 128, 128)
        self.Mixed_4c = Inception(576, 192, 96, 128, 96, 128, 128)
        self.Mixed_4d = Inception(576, 160, 128, 160, 128, 160, 96)
        self.Mixed_4e = Inception(576, 96, 128, 192, 160, 192, 96)
        self.Mixed_5a = Inception_2(576, 128, 192, 192, 256)
        self.Mixed_5b = Inception(1024, 352, 192, 320, 160, 224, 128)
        self.Mixed_5c = Inception(1024, 352, 192, 320, 192, 224, 128)

    def construct(self, inputs):
        """forward function."""
        end_points = {}
        temp_point = 'Conv2d_1a_7x7'
        net = self.Conv2d_1a_7x7(inputs)
        end_points[temp_point] = net
        temp_point = 'MaxPool_2a_3x3'
        net = self.MaxPool_2a_3x3(net)
        end_points[temp_point] = net
        temp_point = 'Conv2d_2b_1x1'
        net = self.Conv2d_2b_1x1(net)
        end_points[temp_point] = net
        temp_point = 'Conv2d_2c_3x3'
        net = self.Conv2d_2c_3x3(net)
        end_points[temp_point] = net
        temp_point = 'MaxPool_3a_3x3'
        net = self.MaxPool_3a_3x3(net)
        end_points[temp_point] = net
        temp_point = 'Mixed_3b'
        net = self.Mixed_3b(net)
        end_points[temp_point] = net
        temp_point = 'Mixed_3c'
        net = self.Mixed_3c(net)
        end_points[temp_point] = net
        # 38 x 38 x 320
        temp_point = 'Mixed_4a'
        net = self.Mixed_4a(net)
        end_points[temp_point] = net
        # 19 x 19 x 576
        temp_point = 'Mixed_4b'
        net = self.Mixed_4b(net)
        end_points[temp_point] = net
        # 19 x 19 x 576
        temp_point = 'Mixed_4c'
        net = self.Mixed_4c(net)
        end_points[temp_point] = net
        # 19 x 19 x 576
        temp_point = 'Mixed_4d'
        net = self.Mixed_4d(net)
        end_points[temp_point] = net
        # 19 x 19 x 576
        temp_point = 'Mixed_4e'
        net = self.Mixed_4e(net)
        end_points[temp_point] = net
        # 19 x 19 x 576
        temp_point = 'Mixed_5a'
        net = self.Mixed_5a(net)
        end_points[temp_point] = net
        temp_point = 'Mixed_5b'
        net = self.Mixed_5b(net)
        end_points[temp_point] = net
        temp_point = 'Mixed_5c'
        net = self.Mixed_5c(net)
        end_points[temp_point] = net
        return end_points
