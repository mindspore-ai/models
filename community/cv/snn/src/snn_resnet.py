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
from mindspore import Tensor
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, HeUniform
from src.ifnode import IFNode


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


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.ifnode1 = IFNode()

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)
        self.ifnode2 = IFNode()

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)

        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

        self.ifnode3 = IFNode()

    def construct(self, x_in):
        """ResidualBlock"""
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


class ResNet_SNN(nn.Cell):
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
        >>> ResNet_SNN(ResidualBlock,
        >>>            [3, 4, 6, 3],
        >>>            [64, 256, 512, 1024],
        >>>            [256, 512, 1024, 2048],
        >>>            [1, 2, 2, 2],
        >>>            10)
    """

    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes):
        super(ResNet_SNN, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.T = 5
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.ifnode1 = IFNode()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer_nums = layer_nums
        # layer_nums:[3, 4, 6, 3]
        self.layer1 = self.make_layer(block, layer_nums[0], in_channel=in_channels[0],
                                      out_channel=out_channels[0], stride=strides[0])
        self.layer2 = self.make_layer(block, layer_nums[1], in_channel=in_channels[1],
                                      out_channel=out_channels[1], stride=strides[1])
        self.layer3 = self.make_layer(block, layer_nums[2], in_channel=in_channels[2],
                                      out_channel=out_channels[2], stride=strides[2])
        self.layer4 = self.make_layer(block, layer_nums[3], in_channel=in_channels[3],
                                      out_channel=out_channels[3], stride=strides[3])

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)
        self.end_ifnode = IFNode(fire=False)
        self.layers = nn.CellList([self.layer1, self.layer2, self.layer3, self.layer4])


    def make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.layer.CellList(layers)


    def construct(self, x_in):
        """ResNet SNN block with graph mode"""
        out = x_in
        v1 = v_end = 0.0

        V = []
        for layer_num in self.layer_nums:
            for _ in range(layer_num):
                V.append([Tensor(0.0), Tensor(0.0), Tensor(0.0)])

        for _ in range(self.T):
            x = self.conv1(x_in)
            x = self.bn1(x)
            x, v1 = self.ifnode1(x, v1)

            c1 = self.maxpool(x)
            out = c1

            index = 0
            ifnode_count = 0
            for row in self.layer_nums:
                layers = self.layers[index]
                for col in range(row):
                    block = layers[col]
                    out, V[ifnode_count + col][0], V[ifnode_count + col][1], V[ifnode_count + col][2] = \
                        block((out, V[ifnode_count + col][0], V[ifnode_count + col][1], V[ifnode_count + col][2]))
                ifnode_count += self.layer_nums[index]
                index += 1

            out = self.mean(out, (2, 3))
            out = self.flatten(out)
            out = self.end_point(out)
            out, v_end = self.end_ifnode(out, v_end)

        return out / self.T


def snn_resnet50(class_num=10):
    return ResNet_SNN(ResidualBlock,
                      [3, 4, 6, 3],
                      [64, 256, 512, 1024],
                      [256, 512, 1024, 2048],
                      [1, 2, 2, 2],
                      class_num)
