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
"""BNInception implementation"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal


class Conv2dBlock(nn.Cell):
    """Conv2dBlock."""

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, pad_mode="pad", has_bias=True,
                 weight_init="he_uniform", bias_init="normal",
                 bn_eps=1e-05, bn_momentum=0.9, frozen_bn=False):
        super().__init__(False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, has_bias=has_bias,
                              weight_init=weight_init, bias_init=bias_init)

        if frozen_bn:
            use_batch_statistics = False
            affine = False
        else:
            # Using the default behaviour
            use_batch_statistics = None
            affine = True

        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum,
                                 use_batch_statistics=use_batch_statistics, affine=affine)

        self.relu = nn.ReLU()

    def construct(self, x):
        """Feed forward"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Cell):
    """
    Inception Block
    """

    def __init__(self, in_channels, n1x1, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2, pool_planes, frozen_bn=False):
        super().__init__()
        self.branch_1x1 = Conv2dBlock(in_channels, n1x1, frozen_bn=frozen_bn)
        self.branch_3x3_reduce = Conv2dBlock(in_channels, n3x3red_a, frozen_bn=frozen_bn)
        self.branch_3x3 = Conv2dBlock(n3x3red_a, n3x3, kernel_size=3, padding=1, frozen_bn=frozen_bn)

        self.branch_double_3x3_reduce = Conv2dBlock(in_channels, n3x3red_b, frozen_bn=frozen_bn)
        self.branch_double_3x3_1 = Conv2dBlock(n3x3red_b, n3x3red_b_2, kernel_size=3, padding=1, frozen_bn=frozen_bn)
        self.branch_double_3x3_2 = Conv2dBlock(n3x3red_b_2, n3x3red_b_2, kernel_size=3, padding=1, frozen_bn=frozen_bn)

        self.pool = nn.SequentialCell([
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1))),
            nn.AvgPool2d(3, 1, pad_mode='valid')
        ])

        self.branch_pool_proj = Conv2dBlock(in_channels, pool_planes, kernel_size=1, frozen_bn=frozen_bn)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """Feed forward"""
        x0 = self.branch_1x1(x)

        x1 = self.branch_3x3_reduce(x)
        x1 = self.branch_3x3(x1)

        x2 = self.branch_double_3x3_reduce(x)
        x2 = self.branch_double_3x3_1(x2)
        x2 = self.branch_double_3x3_2(x2)

        x_pool = self.pool(x)

        x3 = self.branch_pool_proj(x_pool)
        out = self.concat((x0, x1, x2, x3))
        return out


class Inception_small(nn.Cell):
    """
    Inception Small Block
    """

    def __init__(self, in_channels, n3x3red_a, n3x3, n3x3red_b, n3x3red_b_2, frozen_bn=False):
        super().__init__()
        self.branch_3x3_reduce = Conv2dBlock(in_channels, n3x3red_a, frozen_bn=frozen_bn)
        self.branch_3x3 = Conv2dBlock(n3x3red_a, n3x3, kernel_size=3,
                                      stride=2, padding=1, frozen_bn=frozen_bn)

        self.branch_double_3x3_reduce = Conv2dBlock(in_channels, n3x3red_b, frozen_bn=frozen_bn)
        self.branch_double_3x3_1 = Conv2dBlock(n3x3red_b, n3x3red_b_2,
                                               kernel_size=3, padding=1, frozen_bn=frozen_bn)
        self.branch_double_3x3_2 = Conv2dBlock(n3x3red_b_2, n3x3red_b_2,
                                               kernel_size=3, padding=1, stride=2, frozen_bn=frozen_bn)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """Feed forward"""
        x0 = self.branch_3x3_reduce(x)
        x0 = self.branch_3x3(x0)

        x1 = self.branch_double_3x3_reduce(x)
        x1 = self.branch_double_3x3_1(x1)
        x1 = self.branch_double_3x3_2(x1)

        x_pool = self.pool(x)
        out = self.concat((x0, x1, x_pool))
        return out


class BNInception(nn.Cell):
    """Inception BatchNorm"""

    def __init__(self, input_channels=3, out_channels=1000, dropout: int = None, frozen_bn=False):
        super().__init__()

        # frozen_bn always False
        self.conv1_7x7_s2 = Conv2dBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, frozen_bn=False)

        self.max_pool_2a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.conv2_3x3_reduce = Conv2dBlock(64, 64, frozen_bn=frozen_bn)
        self.conv2_3x3 = Conv2dBlock(64, 192, kernel_size=3, padding=1, frozen_bn=frozen_bn)

        self.max_pool_3a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.inception_3a = Inception(192, 64, 64, 64, 64, 96, 32, frozen_bn=frozen_bn)
        self.inception_3b = Inception(256, 64, 64, 96, 64, 96, 64, frozen_bn=frozen_bn)
        self.inception_3c = Inception_small(320, 128, 160, 64, 96, frozen_bn=frozen_bn)

        self.inception_4a = Inception(576, 224, 64, 96, 96, 128, 128, frozen_bn=frozen_bn)
        self.inception_4b = Inception(576, 192, 96, 128, 96, 128, 128, frozen_bn=frozen_bn)
        self.inception_4c = Inception(576, 160, 128, 160, 128, 160, 128, frozen_bn=frozen_bn)
        self.inception_4d = Inception(608, 96, 128, 192, 160, 192, 128, frozen_bn=frozen_bn)
        self.inception_4e = Inception_small(608, 128, 192, 192, 256, frozen_bn=frozen_bn)

        self.inception_5a = Inception(1056, 352, 192, 320, 160, 224, 128, frozen_bn=frozen_bn)
        self.inception_5b = Inception(1024, 352, 192, 320, 192, 224, 128, frozen_bn=frozen_bn)
        self.inception_5b.pool = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same")

        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode="valid")
        self.flatten = nn.Flatten()

        self.use_dropout = dropout is not None and dropout > 0
        if self.use_dropout:
            self.dropout_cell = nn.Dropout(dropout)

        self.fc = nn.Dense(in_channels=1024, out_channels=out_channels, weight_init=Normal(0, 0.001))

    def construct(self, x):
        """Feed forward"""
        x = self.conv1_7x7_s2(x)
        x = self.max_pool_2a_3x3(x)
        x = self.conv2_3x3_reduce(x)
        x = self.conv2_3x3(x)
        x = self.max_pool_3a_3x3(x)

        # inception blocks
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)

        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.global_pool(x)
        x = self.flatten(x)

        if self.use_dropout:
            x = self.dropout_cell(x)

        x = self.fc(x)
        return x
