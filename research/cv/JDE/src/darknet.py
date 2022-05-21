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
"""DarkNet model."""
from mindspore import nn
from mindspore.ops import operations as P


def conv_block(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
):
    """
    Set a conv2d, BN and relu layer.
    """
    pad_mode = 'same'
    padding = 0

    dbl = nn.SequentialCell(
        [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                pad_mode=pad_mode,
            ),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(),
        ]
    )

    return dbl


class ResidualBlock(nn.Cell):
    """
    DarkNet V1 residual block definition.

    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.

    Returns:
        out (ms.Tensor): Output tensor.

    Examples:
        ResidualBlock(3, 32)
    """
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        out_chls = out_channels//2
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


class DarkNet(nn.Cell):
    """
    DarkNet V1 network.

    Args:
        block (cell): Block for network.
        layer_nums (list): Numbers of different layers.
        in_channels (list): Input channel.
        out_channels (list): Output channel.
        detect (bool): Whether detect or not. Default:False.

    Returns:
        if detect = True:
            c11 (ms.Tensor): Output from last layer.

        if detect = False:
            c7, c9, c11 (ms.Tensor): Outputs from different layers (FPN).

    Examples:
        DarkNet(
        ResidualBlock,
        [1, 2, 8, 8, 4],
        [32, 64, 128, 256, 512],
        [64, 128, 256, 512, 1024],
        )
    """
    def __init__(
            self,
            block,
            layer_nums,
            in_channels,
            out_channels,
            detect=False,
    ):
        super().__init__()

        self.detect = detect

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 5:
            raise ValueError("the length of layer_num, inchannel, outchannel list must be 5!")

        self.conv0 = conv_block(
            3,
            in_channels[0],
            kernel_size=3,
            stride=1,
        )

        self.conv1 = conv_block(
            in_channels[0],
            out_channels[0],
            kernel_size=3,
            stride=2,
        )

        self.layer1 = self._make_layer(
            block,
            layer_nums[0],
            in_channel=out_channels[0],
            out_channel=out_channels[0],
        )

        self.conv2 = conv_block(
            in_channels[1],
            out_channels[1],
            kernel_size=3,
            stride=2,
        )

        self.layer2 = self._make_layer(
            block,
            layer_nums[1],
            in_channel=out_channels[1],
            out_channel=out_channels[1],
        )

        self.conv3 = conv_block(
            in_channels[2],
            out_channels[2],
            kernel_size=3,
            stride=2,
        )

        self.layer3 = self._make_layer(
            block,
            layer_nums[2],
            in_channel=out_channels[2],
            out_channel=out_channels[2],
        )

        self.conv4 = conv_block(
            in_channels[3],
            out_channels[3],
            kernel_size=3,
            stride=2,
        )

        self.layer4 = self._make_layer(
            block,
            layer_nums[3],
            in_channel=out_channels[3],
            out_channel=out_channels[3],
        )

        self.conv5 = conv_block(
            in_channels[4],
            out_channels[4],
            kernel_size=3,
            stride=2,
        )

        self.layer5 = self._make_layer(
            block,
            layer_nums[4],
            in_channel=out_channels[4],
            out_channel=out_channels[4],
        )

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        """
        Make Layer for DarkNet.

        Args:
            block (Cell): DarkNet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.

        Examples:
            _make_layer(ConvBlock, 1, 128, 256)
        """
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        Feed forward image.
        """
        c1 = self.conv0(x)
        c2 = self.conv1(c1)
        c3 = self.layer1(c2)
        c4 = self.conv2(c3)
        c5 = self.layer2(c4)
        c6 = self.conv3(c5)
        c7 = self.layer3(c6)
        c8 = self.conv4(c7)
        c9 = self.layer4(c8)
        c10 = self.conv5(c9)
        c11 = self.layer5(c10)

        if self.detect:
            return c7, c9, c11

        return c11


def darknet53():
    """
    Get DarkNet53 neural network.

    Returns:
        Cell, cell instance of DarkNet53 neural network.

    Examples:
        darknet53()
    """

    darknet = DarkNet(
        block=ResidualBlock,
        layer_nums=[1, 2, 8, 8, 4],
        in_channels=[32, 64, 128, 256, 512],
        out_channels=[64, 128, 256, 512, 1024],
    )

    return darknet
