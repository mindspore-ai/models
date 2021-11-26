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
"""Inception-v2 model definition"""
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform, TruncatedNormal
from mindspore.ops import operations as P


class BasicConv2d(nn.Cell):
    """ Basic convolution block for InceptionV2 model. Consist of Conv2d+BN+ReLU6"""
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, pad_mode='same', has_bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, weight_init=XavierUniform(), has_bias=has_bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU6()

    def construct(self, x):
        """Construct a forward graph"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWSConv(nn.Cell):
    """Depthwise separable convolution block for InceptionV2 model"""
    def __init__(self, in_channels, out_channels, depthwise_multiplier,
                 kernel_size=1, stride=1, padding=0, pad_mode="same", stddev=1.0):
        super(DWSConv, self).__init__()
        temp_channel = in_channels * depthwise_multiplier
        self.is_use_pointwise = out_channels
        if out_channels is not None:
            self.conv1 = nn.Conv2d(in_channels, temp_channel, kernel_size,
                                   stride, padding=padding, pad_mode=pad_mode,
                                   group=in_channels, weight_init=TruncatedNormal(stddev))
            self.conv2 = nn.Conv2d(temp_channel, out_channels, kernel_size=1, stride=1, padding=0,
                                   weight_init=TruncatedNormal(stddev))
        else:
            self.conv1 = nn.Conv2d(in_channels, temp_channel, kernel_size, stride, group=in_channels,
                                   padding=padding, pad_mode=pad_mode,
                                   weight_init=TruncatedNormal(stddev))
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU6()

    def construct(self, x):
        """Construct a forward graph"""
        if self.is_use_pointwise is not None:
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionBlock4Branches(nn.Cell):
    """InceptionV2 block with 4 brunches"""
    def __init__(self, in_channels, n1x1, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2, pool_planes):
        super(InceptionBlock4Branches, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, n1x1, kernel_size=1)
        self.branch_1 = nn.SequentialCell([BasicConv2d(in_channels, n3x3red_a, kernel_size=1),
                                           BasicConv2d(n3x3red_a, n3x3, kernel_size=3)])
        self.branch_2 = nn.SequentialCell([BasicConv2d(in_channels, n3x3red_b, kernel_size=1),
                                           BasicConv2d(n3x3red_b, n3x3red_b_2, kernel_size=3),
                                           BasicConv2d(n3x3red_b_2, n3x3red_b_2, kernel_size=3)])
        self.avgpool_op = P.AvgPool(pad_mode="same", kernel_size=3, strides=1)
        self.branch_3 = BasicConv2d(in_channels, pool_planes, kernel_size=1)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """Construct a forward graph"""
        branch_0_x = self.branch_0(x)
        branch_1_x = self.branch_1(x)
        branch_2_x = self.branch_2(x)

        branch_3_x = self.avgpool_op(x)
        branch_3_x = self.branch_3(branch_3_x)

        return self.concat((branch_0_x, branch_1_x, branch_2_x, branch_3_x))


class InceptionBlock3Branches(nn.Cell):
    """InceptionV2 block with 3 branches"""
    def __init__(self, in_channels, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2):
        super(InceptionBlock3Branches, self).__init__()
        self.branch_0 = nn.SequentialCell([BasicConv2d(in_channels, n3x3red_a, kernel_size=1),
                                           BasicConv2d(n3x3red_a, n3x3, kernel_size=3, stride=2)])

        self.branch_1 = nn.SequentialCell([BasicConv2d(in_channels, n3x3red_b, kernel_size=1),
                                           BasicConv2d(n3x3red_b, n3x3red_b_2, kernel_size=3),
                                           BasicConv2d(n3x3red_b_2, n3x3red_b_2, kernel_size=3, stride=2)])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """Construct a forward graph"""
        branch_0_x = self.branch_0(x)
        branch_1_x = self.branch_1(x)
        branch_2_x = self.maxpool(x)
        return self.concat((branch_0_x, branch_1_x, branch_2_x))


class InceptionV2(nn.Cell):
    """Create InceptionV2 model from blocks above"""
    def __init__(self, use_separable_conv=False):
        super(InceptionV2, self).__init__()
        self.feature_map_channels = {'Conv2d_1a_7x7': 64, 'MaxPool_2a_3x3': 64,
                                     'Conv2d_2b_1x1': 64, 'Conv2d_2c_3x3': 192,
                                     'MaxPool_3a_3x3': 192, 'Mixed_3b': 256,
                                     'Mixed_3c': 320, 'Mixed_4a': 576, 'Mixed_4b': 576,
                                     'Mixed_4c': 576, 'Mixed_4d': 576, 'Mixed_4e': 576,
                                     'Mixed_5a': 1024, 'Mixed_5b': 1024, 'Mixed_5c': 1024}
        if use_separable_conv:
            depthwise_multiplier = min(int(64 / 3), 8)
            self.Conv2d_1a_7x7 = DWSConv(3, 64,
                                         depthwise_multiplier=depthwise_multiplier,
                                         kernel_size=7, stride=2, padding=0)
        else:
            self.Conv2d_1a_7x7 = BasicConv2d(3, 64, kernel_size=7, stride=2)
        self.MaxPool_2a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.Conv2d_2b_1x1 = BasicConv2d(64, 64, kernel_size=1)
        self.Conv2d_2c_3x3 = BasicConv2d(64, 192, kernel_size=3)
        self.MaxPool_3a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.Mixed_3b = InceptionBlock4Branches(192, 64, 64, 64, 64, 96, 32)
        self.Mixed_3c = InceptionBlock4Branches(256, 64, 64, 96, 64, 96, 64)
        self.Mixed_4a = InceptionBlock3Branches(320, 128, 160, 64, 96)
        self.Mixed_4b = InceptionBlock4Branches(576, 224, 64, 96, 96, 128, 128)
        self.Mixed_4c = InceptionBlock4Branches(576, 192, 96, 128, 96, 128, 128)
        self.Mixed_4d = InceptionBlock4Branches(576, 160, 128, 160, 128, 160, 96)
        self.Mixed_4e = InceptionBlock4Branches(576, 96, 128, 192, 160, 192, 96)
        self.Mixed_5a = InceptionBlock3Branches(576, 128, 192, 192, 256)
        self.Mixed_5b = InceptionBlock4Branches(1024, 352, 192, 320, 160, 224, 128)
        self.Mixed_5c = InceptionBlock4Branches(1024, 352, 192, 320, 192, 224, 128)

    def construct(self, x):
        """Construct a forward graph"""
        end_points = {}
        temp_point = 'Conv2d_1a_7x7'
        x = self.Conv2d_1a_7x7(x)
        end_points[temp_point] = x
        temp_point = 'MaxPool_2a_3x3'
        x = self.MaxPool_2a_3x3(x)
        end_points[temp_point] = x
        temp_point = 'Conv2d_2b_1x1'
        x = self.Conv2d_2b_1x1(x)
        end_points[temp_point] = x
        temp_point = 'Conv2d_2c_3x3'
        x = self.Conv2d_2c_3x3(x)
        end_points[temp_point] = x
        temp_point = 'MaxPool_3a_3x3'
        x = self.MaxPool_3a_3x3(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_3b'
        x = self.Mixed_3b(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_3c'
        x = self.Mixed_3c(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_4a'
        x = self.Mixed_4a(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_4b'
        x = self.Mixed_4b(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_4c'
        x = self.Mixed_4c(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_4d'
        x = self.Mixed_4d(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_4e'
        x = self.Mixed_4e(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_5a'
        x = self.Mixed_5a(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_5b'
        x = self.Mixed_5b(x)
        end_points[temp_point] = x
        temp_point = 'Mixed_5c'
        x = self.Mixed_5c(x)
        end_points[temp_point] = x
        return end_points


def inception_v2():
    """Build an InceptionV2 network"""
    return InceptionV2()
