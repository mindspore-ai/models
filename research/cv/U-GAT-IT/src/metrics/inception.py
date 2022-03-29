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
"""
InceptionV1 for metrics
"""
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P


def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)


class Conv2dBlock(nn.Cell):
    """
     Basic convolutional block
     Args:
         in_channles (int): Input channel.
         out_channels (int): Output channel.
         kernel_size (int): Input kernel size. Default: 1
         stride (int): Stride size for the first convolutional layer. Default: 1.
         padding (int): Implicit paddings on both sides of the input. Default: 0.
         pad_mode (str): Padding mode. Optional values are "same", "valid", "pad". Default: "same".
      Returns:
          Tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="same"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, weight_init=weight_variable())
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class InceptionBlockV1(nn.Cell):
    """
    Inception Block V1
    """

    def __init__(self, in_channels, n1, n2_0, n2_1, n3_0, n3_1, n3_2, n4):
        super().__init__()
        self.conv = Conv2dBlock(in_channels, n1, kernel_size=1, pad_mode="same")

        self.tower_conv = Conv2dBlock(in_channels, n2_0, kernel_size=1, pad_mode="same")
        self.tower_conv_1 = Conv2dBlock(n2_0, n2_1, kernel_size=5, pad_mode="same")

        self.tower_1_conv = Conv2dBlock(in_channels, n3_0, kernel_size=1, pad_mode="same")
        self.tower_1_conv_1 = Conv2dBlock(n3_0, n3_1, kernel_size=3, pad_mode="same")
        self.tower_1_conv_2 = Conv2dBlock(n3_1, n3_2, kernel_size=3, pad_mode="same")

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="same")
        self.tower_2_conv = Conv2dBlock(in_channels, n4, kernel_size=1, pad_mode="same")
        self.join = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.conv(x)

        branch2 = self.tower_conv(x)
        branch2 = self.tower_conv_1(branch2)

        branch3 = self.tower_1_conv(x)
        branch3 = self.tower_1_conv_1(branch3)
        branch3 = self.tower_1_conv_2(branch3)

        cell = self.avgpool(x)
        branch4 = self.tower_2_conv(cell)
        return self.join((branch1, branch2, branch3, branch4))


class InceptionBlockV2(nn.Cell):
    """
    Inception Block V2
    """

    def __init__(self, in_channels, n1, n2_0, n2_1, n2_2):
        super().__init__()
        self.conv = Conv2dBlock(in_channels, n1, kernel_size=3, stride=2, pad_mode="valid")

        self.tower_conv = Conv2dBlock(in_channels, n2_0, kernel_size=1, pad_mode="same")
        self.tower_conv_1 = Conv2dBlock(n2_0, n2_1, kernel_size=3, pad_mode="same")
        self.tower_conv_2 = Conv2dBlock(n2_1, n2_2, kernel_size=3, stride=2, pad_mode="valid")

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        self.join = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.conv(x)

        branch2 = self.tower_conv(x)
        branch2 = self.tower_conv_1(branch2)
        branch2 = self.tower_conv_2(branch2)

        branch3 = self.maxpool(x)
        return self.join((branch1, branch2, branch3))


class InceptionBlockV3(nn.Cell):
    """
    Inception Block V3
    """

    def __init__(self, in_channels, n_interm, n_concat):
        super().__init__()
        self.conv = Conv2dBlock(in_channels, n_concat, kernel_size=1)

        self.tower_conv = Conv2dBlock(in_channels, n_interm, kernel_size=1)
        self.tower_conv_1 = Conv2dBlock(n_interm, n_interm, kernel_size=(1, 7))
        self.tower_conv_2 = Conv2dBlock(n_interm, n_concat, kernel_size=(7, 1))

        self.tower_1_conv = Conv2dBlock(in_channels, n_interm, kernel_size=1)
        self.tower_1_conv_1 = Conv2dBlock(n_interm, n_interm, kernel_size=(7, 1))
        self.tower_1_conv_2 = Conv2dBlock(n_interm, n_interm, kernel_size=(1, 7))
        self.tower_1_conv_3 = Conv2dBlock(n_interm, n_interm, kernel_size=(7, 1))
        self.tower_1_conv_4 = Conv2dBlock(n_interm, n_concat, kernel_size=(1, 7))

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="same")
        self.tower_2_conv = Conv2dBlock(in_channels, n_concat, kernel_size=1)
        self.join = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.conv(x)

        branch2 = self.tower_conv(x)
        branch2 = self.tower_conv_1(branch2)
        branch2 = self.tower_conv_2(branch2)

        branch3 = self.tower_1_conv(x)
        branch3 = self.tower_1_conv_1(branch3)
        branch3 = self.tower_1_conv_2(branch3)
        branch3 = self.tower_1_conv_3(branch3)
        branch3 = self.tower_1_conv_4(branch3)

        cell = self.avgpool(x)
        branch4 = self.tower_2_conv(cell)
        return self.join((branch1, branch2, branch3, branch4))


class InceptionBlockV4(nn.Cell):
    """
    Inception Block V4
    """

    def __init__(self, in_channels, n_interm, n_left):
        super().__init__()
        self.tower_conv = Conv2dBlock(in_channels, n_interm, kernel_size=1, pad_mode="same")
        self.tower_conv_1 = Conv2dBlock(n_interm, n_left, kernel_size=3, stride=2, pad_mode="valid")

        self.tower_1_conv = Conv2dBlock(in_channels, n_interm, kernel_size=1, pad_mode="same")
        self.tower_1_conv_1 = Conv2dBlock(n_interm, n_interm, kernel_size=(1, 7), pad_mode="same")
        self.tower_1_conv_2 = Conv2dBlock(n_interm, n_interm, kernel_size=(7, 1), pad_mode="same")
        self.tower_1_conv_3 = Conv2dBlock(n_interm, n_interm, kernel_size=3, stride=2, pad_mode="valid")

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        self.join = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.tower_conv(x)
        branch1 = self.tower_conv_1(branch1)

        branch2 = self.tower_1_conv(x)
        branch2 = self.tower_1_conv_1(branch2)
        branch2 = self.tower_1_conv_2(branch2)
        branch2 = self.tower_1_conv_3(branch2)

        branch3 = self.maxpool(x)

        return self.join((branch1, branch2, branch3))


class InceptionBlockV5(nn.Cell):
    """
    Inception Block V5
    """

    def __init__(self, in_channels, n1_left, n1_right, n3x3, n5x5, maxpool=True):
        super().__init__()
        self.conv = Conv2dBlock(in_channels, n1_left, kernel_size=1, pad_mode="same")

        self.tower_conv = Conv2dBlock(in_channels, n3x3, kernel_size=1, pad_mode="same")
        self.tower_mixed_conv = Conv2dBlock(n3x3, n3x3, kernel_size=(1, 3), pad_mode="same")
        self.tower_mixed_conv_1 = Conv2dBlock(n3x3, n3x3, kernel_size=(3, 1), pad_mode="same")

        self.tower_1_conv = Conv2dBlock(in_channels, n5x5, kernel_size=1, pad_mode="same")
        self.tower_1_conv_1 = Conv2dBlock(n5x5, n3x3, kernel_size=3, pad_mode="same")
        self.tower_1_mixed_conv = Conv2dBlock(n3x3, n3x3, kernel_size=(1, 3), pad_mode="same")
        self.tower_1_mixed_conv_1 = Conv2dBlock(n3x3, n3x3, kernel_size=(3, 1), pad_mode="same")

        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same")
        else:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="same")

        self.tower_2_conv = Conv2dBlock(in_channels, n1_right, kernel_size=1, pad_mode="same")
        self.join = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.conv(x)

        branch2 = self.tower_conv(x)
        branch2_left = self.tower_mixed_conv(branch2)
        branch2_right = self.tower_mixed_conv_1(branch2)
        branch2 = self.join((branch2_left, branch2_right))

        branch3 = self.tower_1_conv(x)
        branch3 = self.tower_1_conv_1(branch3)
        branch3_left = self.tower_1_mixed_conv(branch3)
        branch3_right = self.tower_1_mixed_conv_1(branch3)
        branch3 = self.join((branch3_left, branch3_right))

        cell = self.pool(x)
        branch4 = self.tower_2_conv(cell)

        return self.join((branch1, branch2, branch3, branch4))


class InceptionForDistance(nn.Cell):
    """
    InceptionV1 architecture
    """

    def __init__(self):
        super().__init__()

        self.conv = Conv2dBlock(3, 32, kernel_size=3, stride=2, pad_mode="valid")
        self.conv_1 = Conv2dBlock(32, 32, kernel_size=3, stride=1, pad_mode="valid")
        self.conv_2 = Conv2dBlock(32, 64, kernel_size=3, stride=1, pad_mode="same")
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.conv_3 = Conv2dBlock(64, 80, kernel_size=1, stride=1, pad_mode="valid")
        self.conv_4 = Conv2dBlock(80, 192, kernel_size=3, stride=1, pad_mode="valid")
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.mixed = InceptionBlockV1(192, 64, 48, 64, 64, 96, 96, 32)
        self.mixed_1 = InceptionBlockV1(256, 64, 48, 64, 64, 96, 96, 64)
        self.mixed_2 = InceptionBlockV1(288, 64, 48, 64, 64, 96, 96, 64)

        self.mixed_3 = InceptionBlockV2(288, 384, 64, 96, 96)
        self.mixed_4 = InceptionBlockV3(768, 128, 192)
        self.mixed_5 = InceptionBlockV3(768, 160, 192)
        self.mixed_6 = InceptionBlockV3(768, 160, 192)
        self.mixed_7 = InceptionBlockV3(768, 192, 192)

        self.mixed_8 = InceptionBlockV4(768, 192, 320)

        self.mixed_9 = InceptionBlockV5(1280, 320, 192, 384, 448, maxpool=False)
        self.mixed_10 = InceptionBlockV5(2048, 320, 192, 384, 448, maxpool=True)

    def construct(self, x):
        """construct"""
        x = self.conv(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.maxpool1(x)

        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.maxpool2(x)

        x = self.mixed(x)
        x = self.mixed_1(x)
        x = self.mixed_2(x)
        x = self.mixed_3(x)
        x = self.mixed_4(x)
        x = self.mixed_5(x)
        x = self.mixed_6(x)
        x = self.mixed_7(x)
        x = self.mixed_8(x)
        x = self.mixed_9(x)
        x = self.mixed_10(x)

        return x
