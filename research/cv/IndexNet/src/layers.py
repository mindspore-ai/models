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
"""Model layers."""
from mindspore import nn


def depth_sep_dilated_conv_3x3_bn(inp, oup, padding, dilation):
    """
    Dilated depthwise separable convolution block with BN, ReLU.

    Args:
        inp (int): Input channels of block.
        oup (int): Output channels of block.
        padding (int): Padding of depthwise conv.
        dilation (int): Dilation of depthwise conv.

    Returns:
        block: Dilated depthwise separable conv block.
    """
    return nn.SequentialCell(
        [
            nn.Conv2d(
                in_channels=inp,
                out_channels=inp,
                kernel_size=3,
                stride=1,
                pad_mode='pad',
                padding=padding,
                dilation=dilation,
                group=inp,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=inp),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=inp,
                out_channels=oup,
                kernel_size=1,
                stride=1,
                pad_mode='pad',
                padding=0,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=oup),
            nn.ReLU6()
        ]
    )


def dep_sep_conv_bn(inp, oup, k, s):
    """
    Depthwise separable convolution block with BN, ReLU.

    Args:
        inp (int): Input channels of block.
        oup (int): Output channels of block.
        k (int): Kernel size of depthwise conv.
        s (int): Stride of depthwise conv.

    Returns:
        block: Depthwise separable conv block.
    """
    return nn.SequentialCell(
        [
            nn.Conv2d(
                in_channels=inp,
                out_channels=inp,
                kernel_size=k,
                stride=s,
                pad_mode='pad',
                padding=k // 2,
                group=inp,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=inp),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=inp,
                out_channels=oup,
                kernel_size=1,
                stride=1,
                pad_mode='pad',
                padding=0,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=oup),
            nn.ReLU6()
        ]
    )


def conv_bn(inp, oup, k, s):
    """
    Conv, BN, ReLU block.

    Args:
        inp (int): Input channels of block.
        oup (int): Output channels of block.
        k (int): Kernel size of conv.
        s (int): Stride of conv.

    Returns:
        block: Conv, BN, activation block.
    """
    return nn.SequentialCell(
        [
            nn.Conv2d(
                in_channels=inp,
                out_channels=oup,
                kernel_size=k,
                stride=s,
                pad_mode='pad',
                padding=k // 2,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=oup),
            nn.ReLU6()
        ]
    )


def pred(inp, oup, conv_operator, k):
    """
    Output conv block.

    Args:
        inp (int): Input channels of block.
        oup (int): Output channels of block.
        conv_operator (str): Type of conv operator to use as input conv block.
        k (int): Kernel size of convs.

    Returns:
        block: Last convs mask prediction block.
    """
    # the last 1x1 convolutional layer is very important
    hlconv2d = hlconv[conv_operator]
    return nn.SequentialCell(
        [
            hlconv2d(inp, oup, k, 1),
            nn.Conv2d(
                in_channels=oup,
                out_channels=oup,
                kernel_size=k,
                stride=1,
                pad_mode='pad',
                padding=k // 2,
                has_bias=False,
            )
        ]
    )


hlconv = {
    'std_conv': conv_bn,
    'dep_sep_conv': dep_sep_conv_bn,
}
