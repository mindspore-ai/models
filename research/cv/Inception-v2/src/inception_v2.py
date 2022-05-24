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
"""inceptionv2 net"""
import mindspore.nn as nn
import mindspore.ops as op
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P


def weight_variable(stddev):
    """Weight variable."""
    return TruncatedNormal(stddev)


class Conv2dBlock(nn.Cell):
    """Conv2dBlock"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 pad_mode="same", is_variable=True):
        super(Conv2dBlock, self).__init__()
        self.is_variable = is_variable
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, pad_mode=pad_mode, weight_init="XavierUniform")
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        x = self.conv_1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class depthwise_separable_conv(nn.Cell):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, pad_mode="same"):
        super(depthwise_separable_conv, self).__init__()
        self.is_use_pointwise = out_channels
        if out_channels is not None:
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding=padding, pad_mode=pad_mode, group=in_channels,
                                   weight_init="XavierUniform")
            self.conv2 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0, weight_init="XavierUniform")
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, group=in_channels,
                                   padding=padding, pad_mode=pad_mode,
                                   weight_init="XavierUniform")
        self.Relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        if self.is_use_pointwise is not None:
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
        x = self.bn(x)
        x = self.Relu(x)
        return x


class Inception(nn.Cell):
    """Inception Block"""

    def __init__(self, in_channels, n1x1, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2, pool_planes):
        super(Inception, self).__init__()
        self.b1 = Conv2dBlock(in_channels, n1x1, kernel_size=1, is_variable=False)
        self.b2 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_a, kernel_size=1),
                                     Conv2dBlock(n3x3red_a, n3x3, kernel_size=3, padding=0, is_variable=False)])
        self.b3 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_b, kernel_size=1),
                                     Conv2dBlock(n3x3red_b, n3x3red_b_2, kernel_size=3, padding=0, is_variable=False),
                                     Conv2dBlock(n3x3red_b_2, n3x3red_b_2, kernel_size=3, padding=0,
                                                 is_variable=False)])
        self.avgpool_op = op.AvgPool(pad_mode="SAME", kernel_size=3, strides=1)
        self.b4 = Conv2dBlock(in_channels, pool_planes, kernel_size=1)
        self.concat = op.Concat(axis=1)

    def construct(self, x):
        """construct"""
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.b3(x)
        cell = self.avgpool_op(x)
        branch4 = self.b4(cell)
        return self.concat((branch1, branch2, branch3, branch4))


class Inception_2(nn.Cell):
    """Inception_2 Block"""

    def __init__(self, in_channels, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2):
        super(Inception_2, self).__init__()
        self.b1 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_a, kernel_size=1),
                                     Conv2dBlock(n3x3red_a, n3x3, kernel_size=3,
                                                 stride=2, padding=0, is_variable=False)])
        self.b2 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red_b, kernel_size=1),
                                     Conv2dBlock(n3x3red_b, n3x3red_b_2, kernel_size=3, padding=0, is_variable=False),
                                     Conv2dBlock(n3x3red_b_2, n3x3red_b_2, kernel_size=3,
                                                 padding=0, stride=2, is_variable=False)])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="SAME")
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """construct"""
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.maxpool(x)
        return self.concat((branch1, branch2, branch3))


class Logits(nn.Cell):
    """Module for Loss"""

    def __init__(self, num_classes=10, dropout_keep_prob=0.8):
        super(Logits, self).__init__()
        self.avg_pool = nn.AvgPool2d(7, pad_mode='valid')
        self.dropout = nn.Dropout(keep_prob=dropout_keep_prob)
        self.flatten = P.Flatten()
        self.fc = nn.Dense(1024, num_classes)

    def construct(self, x):
        """construct"""
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class inception_v2_base(nn.Cell):
    """Detail for net"""

    def __init__(self, num_classes=10, input_channels=3, use_separable_conv=False,
                 dropout_keep_prob=0.8, include_top=True):
        super(inception_v2_base, self).__init__()
        self.feature_map_channels = {'Conv2d_1a_7x7': 64, 'MaxPool_2a_3x3': 64,
                                     'Conv2d_2b_1x1': 64, 'Conv2d_2c_3x3': 192,
                                     'MaxPool_3a_3x3': 192, 'Mixed_3b': 256,
                                     'Mixed_3c': 320, 'Mixed_4a': 576, 'Mixed_4b': 576,
                                     'Mixed_4c': 576, 'Mixed_4d': 576, 'Mixed_4e': 576,
                                     'Mixed_5a': 1024, 'Mixed_5b': 1024, 'Mixed_5c': 1024}

        if use_separable_conv:
            self.Conv2d_1a_7x7 = depthwise_separable_conv(input_channels, 64,
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
        self.include_top = include_top
        if self.include_top:
            self.logits = Logits(num_classes, dropout_keep_prob)

    def construct(self, inputs):
        """inceptionv2 construct"""
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
        # 28 x 28 x 320
        temp_point = 'Mixed_4a'
        net = self.Mixed_4a(net)
        end_points[temp_point] = net
        # 14 x 14 x 576
        temp_point = 'Mixed_4b'
        net = self.Mixed_4b(net)
        end_points[temp_point] = net
        # 14 x 14 x 576
        temp_point = 'Mixed_4c'
        net = self.Mixed_4c(net)
        end_points[temp_point] = net
        # 14 x 14 x 576
        temp_point = 'Mixed_4d'
        net = self.Mixed_4d(net)
        end_points[temp_point] = net
        # 14 x 14 x 576
        temp_point = 'Mixed_4e'
        net = self.Mixed_4e(net)
        end_points[temp_point] = net
        # 14 x 14 x 576
        temp_point = 'Mixed_5a'
        net = self.Mixed_5a(net)
        end_points[temp_point] = net
        temp_point = 'Mixed_5b'
        net = self.Mixed_5b(net)
        end_points[temp_point] = net
        temp_point = 'Mixed_5c'
        net = self.Mixed_5c(net)
        end_points[temp_point] = net
        if not self.include_top:
            return net
        logits = self.logits(net)
        return logits
