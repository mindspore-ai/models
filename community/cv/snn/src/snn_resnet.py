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
"""ResNet_SNN."""
import math
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, HeUniform
from src.ifnode import IFNode_GRAPH, IFNode_PYNATIVE


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same',
                     weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    return nn.Dense(in_channel, out_channel, has_bias=True,
                    weight_init=HeUniform(negative_slope=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'),
                    bias_init=0)


class ResidualBlock_GRAPH(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock_GRAPH(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock_GRAPH, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.ifnode1 = IFNode_GRAPH()

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)
        self.ifnode2 = IFNode_GRAPH()

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

        self.ifnode3 = IFNode_GRAPH()

    def construct(self, x_in):
        """ResidualBlock with graph mode"""
        x, v1, v2, v3 = x_in
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out, v1 = self.ifnode1(out, v1)

        out = self.conv2(out)
        out = self.bn2(out)
        out, v2 = self.ifnode2(out, v2)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out, v3 = self.ifnode3(out, v3)
        return (out, v1, v2, v3)


class ResNet_SNN_GRAPH(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet_SNN_GRAPH(ResidualBlock,
        >>>                  [3, 4, 6, 3],
        >>>                  [64, 256, 512, 1024],
        >>>                  [256, 512, 1024, 2048],
        >>>                  [1, 2, 2, 2],
        >>>                  10)
    """

    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes):
        super(ResNet_SNN_GRAPH, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.T = 5
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.ifnode1 = IFNode_GRAPH()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        # layer_nums:[3, 4, 6, 3]
        self.layer1_1 = self._make_layer_test1(block, in_channel=in_channels[0],
                                               out_channel=out_channels[0], stride=strides[0])
        self.layer1_2 = self._make_layer_test2(block, out_channel=out_channels[0],)
        self.layer1_3 = self._make_layer_test2(block, out_channel=out_channels[0],)
        self.layer2_1 = self._make_layer_test1(block, in_channel=in_channels[1],
                                               out_channel=out_channels[1], stride=strides[1])
        self.layer2_2 = self._make_layer_test2(block, out_channel=out_channels[1])
        self.layer2_3 = self._make_layer_test2(block, out_channel=out_channels[1])
        self.layer2_4 = self._make_layer_test2(block, out_channel=out_channels[1])
        self.layer3_1 = self._make_layer_test1(block, in_channel=in_channels[2],
                                               out_channel=out_channels[2], stride=strides[2])
        self.layer3_2 = self._make_layer_test2(block, out_channel=out_channels[2])
        self.layer3_3 = self._make_layer_test2(block, out_channel=out_channels[2])
        self.layer3_4 = self._make_layer_test2(block, out_channel=out_channels[2])
        self.layer3_5 = self._make_layer_test2(block, out_channel=out_channels[2])
        self.layer3_6 = self._make_layer_test2(block, out_channel=out_channels[2])
        self.layer4_1 = self._make_layer_test1(block, in_channel=in_channels[3],
                                               out_channel=out_channels[3], stride=strides[3])
        self.layer4_2 = self._make_layer_test2(block, out_channel=out_channels[3])
        self.layer4_3 = self._make_layer_test2(block, out_channel=out_channels[3])

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)
        self.end_ifnode = IFNode_GRAPH(fire=False)


    def _make_layer_test1(self, block, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.
        """
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def _make_layer_test2(self, block, out_channel):
        """
            Make stage network of ResNet.

            Args:
                block (Cell): Resnet block.
                out_channel (int): Output channel.
            Returns:
                SequentialCell, the output layer.
            """
        layers = []
        resnet_block = block(out_channel, out_channel, stride=1)
        layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x_in):
        """ResNet SNN block with graph mode"""
        out = x_in
        v1 = v_end = 0.0
        # layer_nums:[3, 4, 6, 3]
        v1_1_1 = v1_1_2 = v1_1_3 = v1_2_1 = v1_2_2 = v1_2_3 = v1_3_1 = v1_3_2 = v1_3_3 = 0.0
        v2_1_1 = v2_1_2 = v2_1_3 = v2_2_1 = v2_2_2 = v2_2_3 = v2_3_1 = v2_3_2 = v2_3_3 = v2_4_1 = v2_4_2 = v2_4_3 = 0.0
        v3_1_1 = v3_1_2 = v3_1_3 = v3_2_1 = v3_2_2 = v3_2_3 = v3_3_1 = v3_3_2 = v3_3_3 = 0.0
        v3_4_1 = v3_4_2 = v3_4_3 = v3_5_1 = v3_5_2 = v3_5_3 = v3_6_1 = v3_6_2 = v3_6_3 = 0.0
        v4_1_1 = v4_1_2 = v4_1_3 = v4_2_1 = v4_2_2 = v4_2_3 = v4_3_1 = v4_3_2 = v4_3_3 = 0.0

        for _ in range(self.T):
            x = self.conv1(x_in)
            x = self.bn1(x)
            x, v1 = self.ifnode1(x, v1)

            c1 = self.maxpool(x)

            c1_1, v1_1_1, v1_1_2, v1_1_3 = self.layer1_1((c1, v1_1_1, v1_1_2, v1_1_3))
            c1_2, v1_2_1, v1_2_2, v1_2_3 = self.layer1_2((c1_1, v1_2_1, v1_2_2, v1_2_3))
            c1_3, v1_3_1, v1_3_2, v1_3_3 = self.layer1_3((c1_2, v1_3_1, v1_3_2, v1_3_3))
            c2_1, v2_1_1, v2_1_2, v2_1_3 = self.layer2_1((c1_3, v2_1_1, v2_1_2, v2_1_3))
            c2_2, v2_2_1, v2_2_2, v2_2_3 = self.layer2_2((c2_1, v2_2_1, v2_2_2, v2_2_3))
            c2_3, v2_3_1, v2_3_2, v2_3_3 = self.layer2_3((c2_2, v2_3_1, v2_3_2, v2_3_3))
            c2_4, v2_4_1, v2_4_2, v2_4_3 = self.layer2_4((c2_3, v2_4_1, v2_4_2, v2_4_3))
            c3_1, v3_1_1, v3_1_2, v3_1_3 = self.layer3_1((c2_4, v3_1_1, v3_1_2, v3_1_3))
            c3_2, v3_2_1, v3_2_2, v3_2_3 = self.layer3_2((c3_1, v3_2_1, v3_2_2, v3_2_3))
            c3_3, v3_3_1, v3_3_2, v3_3_3 = self.layer3_3((c3_2, v3_3_1, v3_3_2, v3_3_3))
            c3_4, v3_4_1, v3_4_2, v3_4_3 = self.layer3_4((c3_3, v3_4_1, v3_4_2, v3_4_3))
            c3_5, v3_5_1, v3_5_2, v3_5_3 = self.layer3_5((c3_4, v3_5_1, v3_5_2, v3_5_3))
            c3_6, v3_6_1, v3_6_2, v3_6_3 = self.layer3_6((c3_5, v3_6_1, v3_6_2, v3_6_3))
            c4_1, v4_1_1, v4_1_2, v4_1_3 = self.layer4_1((c3_6, v4_1_1, v4_1_2, v4_1_3))
            c4_2, v4_2_1, v4_2_2, v4_2_3 = self.layer4_2((c4_1, v4_2_1, v4_2_2, v4_2_3))
            c4_3, v4_3_1, v4_3_2, v4_3_3 = self.layer4_3((c4_2, v4_3_1, v4_3_2, v4_3_3))

            out = self.mean(c4_3, (2, 3))
            out = self.flatten(out)
            out = self.end_point(out)
            out, v_end = self.end_ifnode(out, v_end)

        return out / self.T


class ResidualBlock_PYNATIVE(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock_PYNATIVE(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock_PYNATIVE, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.ifnode1 = IFNode_PYNATIVE()

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)
        self.ifnode2 = IFNode_PYNATIVE()

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

        self.ifnode3 = IFNode_PYNATIVE()

    def construct(self, x):
        """ResidualBlock with pynative mode"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ifnode1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ifnode2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out = out + identity
        out = self.ifnode3(out)

        return out


class ResNet_SNN_PYNATIVE(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet_SNN_PYNATIVE(ResidualBlock,
        >>>                  [3, 4, 6, 3],
        >>>                  [64, 256, 512, 1024],
        >>>                  [256, 512, 1024, 2048],
        >>>                  [1, 2, 2, 2],
        >>>                  10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet_SNN_PYNATIVE, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.T = 5
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.ifnode1 = IFNode_PYNATIVE()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block, layer_nums[0], in_channel=in_channels[0],
                                       out_channel=out_channels[0], stride=strides[0])
        self.layer2 = self._make_layer(block, layer_nums[1], in_channel=in_channels[1],
                                       out_channel=out_channels[1], stride=strides[1])
        self.layer3 = self._make_layer(block, layer_nums[2], in_channel=in_channels[2],
                                       out_channel=out_channels[2], stride=strides[2])
        self.layer4 = self._make_layer(block, layer_nums[3], in_channel=in_channels[3],
                                       out_channel=out_channels[3], stride=strides[3])

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)
        self.end_ifnode = IFNode_PYNATIVE(fire=False)

    def construct(self, x_in):
        """ResNet SNN block with pynative mode"""
        out = x_in
        for _ in range(self.T):
            x = self.conv1(x_in)
            x = self.bn1(x)
            x = self.ifnode1(x)

            c1 = self.maxpool(x)

            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)

            out = self.mean(c5, (2, 3))
            out = self.flatten(out)
            out = self.end_point(out)
            out = self.end_ifnode(out)

        return out / self.T

    def reset_net(self):
        for item in self.cells():
            if isinstance(type(item), type(nn.SequentialCell())):
                if hasattr(item[-1], 'reset'):
                    item[-1].reset()
            else:
                if hasattr(item, 'reset'):
                    item.reset()

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

def snn_resnet50_graph(class_num=10):
    return ResNet_SNN_GRAPH(ResidualBlock_GRAPH,
                            [3, 4, 6, 3],
                            [64, 256, 512, 1024],
                            [256, 512, 1024, 2048],
                            [1, 2, 2, 2],
                            class_num)


def snn_resnet50_pynative(class_num=10):
    return ResNet_SNN_PYNATIVE(ResidualBlock_PYNATIVE,
                               [3, 4, 6, 3],
                               [64, 256, 512, 1024],
                               [256, 512, 1024, 2048],
                               [1, 2, 2, 2],
                               class_num)
