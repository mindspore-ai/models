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
"""
Build resnet34
"""
from typing import List
from typing import Tuple

import mindspore.nn as nn
from mindspore.ops import operations as P


def _conv3x3(
        in_channel: int,
        out_channel: int,
        stride: int = 1,
) -> nn.Conv2d:
    """Construct a convolution layer with 3x3 window.

    Args:
        in_channel (int): A number of the input channels.
        out_channel (int): A number of the output channels
        stride (int): A number that represents the height and width of the kernel movement.

    Returns:
        (nn.Conv2d): A 3x3 Convolution layer.
    """
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        pad_mode='pad',
    )


def _conv1x1(
        in_channel: int,
        out_channel: int,
        stride: int = 1,
) -> nn.Conv2d:
    """Construct a convolution layer with 1x1 window.

    Args:
        in_channel (int): A number of the input channels.
        out_channel (int): A number of the output channels
        stride (int): A number that represents the height and width of the kernel movement.

    Returns:
        (nn.Conv2d): A 1x1 Convolution layer.
    """
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=stride,
        padding=1,
        pad_mode='pad',
    )


def _conv7x7(
        in_channel: int,
        out_channel: int,
        stride: int = 1,
) -> nn.Conv2d:
    """Construct a convolution layer with 7x7 window.

    Args:
        in_channel (int): A number of the input channels.
        out_channel (int): A number of the output channels
        stride (int): A number that represents the height and width of the kernel movement.

    Returns:
        (nn.Conv2d): A 7x7 Convolution layer.
    """
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=7,
        stride=stride,
        padding=3,
        pad_mode='pad',
    )


def _bn(channel: int) -> nn.BatchNorm2d:
    """Construct two dimensional batch-normalization layer.

    Args:
        channel (int): A number of channels.

    Returns:
        (nn.BatchNorm2d): A batch-normalization layer.
    """
    return nn.BatchNorm2d(
        channel,
        eps=1e-5,
        momentum=0.9,
        gamma_init=1,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _modify_conv_stride_dilation(
        conv: nn.Conv2d,
        stride: Tuple[int, int] = (1, 1),
        padding: int = None,
) -> None:
    """Change the stride and the padding value for the convolution layer.

    Args:
        conv (nn.Conv2d): An input convolution layer to modify.
        stride (Tuple[int, int]): A stride value for the convolution.
        padding (int): A padding value for the convolution.

    Returns:
        None
    """
    conv.stride = stride

    if padding is not None:
        conv.padding = padding


def _modify_block(
        block: List[nn.SequentialCell],
        bottleneck: bool = False,
        **kwargs,
) -> None:
    """Modify convolutions in the cells of the given block.

    Args:
        block (List[nn.SequentialCell]): A block to modify.
        bottleneck (bool): Use a bottleneck to reduce the number of parameters and matrix multiplications.
        **kwargs: Kwargs

    Returns:
        None
    """
    for cell in block:
        if bottleneck:
            _modify_conv_stride_dilation(cell.conv2, **kwargs)
        else:
            _modify_conv_stride_dilation(cell.conv1, **kwargs)
        if cell.down_sample_layer is not None:
            # need to make sure no padding for the 1x1 residual connection
            _modify_conv_stride_dilation(list(cell.down_sample_layer)[0], **kwargs)


class BasicBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> BasicBlock(3, 64, stride=2)
    """
    expansion = 1

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv3x3(in_channel, channel, stride=stride)
        self.bn1 = _bn(channel)
        self.relu = nn.ReLU()
        self.conv2 = _conv3x3(channel, channel, stride=1)
        self.bn2 = _bn(channel)

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = \
                nn.SequentialCell([
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size=1,
                        stride=stride,
                        pad_mode='valid',
                    ),
                    _bn(out_channel),
                ])
        self.add = P.Add()

    def construct(self, x):
        """Construct net"""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet34(nn.Cell):
    """
    ResNet34 architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet34(BasicBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 64, 128, 256],
        >>>        [64, 128, 256, 512],
        >>>        [1, 2, 2, 2]),
        >>>        6)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides):
        super(ResNet34, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 3:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 3!")
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])

        _modify_block(list(self.layer3), stride=(1, 1))

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(BasicBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        Forward
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        return [c4]


def resnet34():
    """
    Get ResNet34 neural network.

    Returns:
        Cell, cell instance of ResNet34 neural network.

    Examples:
        >>> net = resnet34()
    """
    return ResNet34(BasicBlock, [3, 4, 6], [64, 64, 128], [64, 128, 256], [1, 2, 1])
