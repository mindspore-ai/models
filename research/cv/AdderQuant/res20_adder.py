# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""ResNet."""
import math
import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import adder_quant


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_planes, out_planes, stride=1):
    # weight_shape = (out_channel, in_channel, 3, 3)
    # weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    # return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)
    return adder_quant.QAdder2dKmeansGroupShare(
        in_planes, out_planes, nbit_a=8, nbit_w=8, quant_group=4, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    # weight_shape = (out_channel, in_channel, 1, 1)
    # weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    # return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, pad_mode='pad', weight_init=weight)
    return adder_quant.QAdder2dKmeansGroupShare(
        in_planes, out_planes, nbit_a=8, nbit_w=8, quant_group=4, kernel_size=1, stride=stride, padding=0, bias=False)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

def _bn1d(channel):
    return nn.BatchNorm1d(channel, eps=1e-5, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
    return nn.Dense(in_channel, out_channel, has_bias=False, weight_init=weight, bias_init=0)


class BasicBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.
    """
    expansion = 1
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = _conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = _bn(out_channel)
        self.conv2 = _conv3x3(out_channel, out_channel, stride=1)
        self.bn2 = _bn(out_channel)
        self.relu = nn.ReLU()

        self.downsample = False
        if stride != 1 or in_channel != out_channel:
            self.downsample = True

        self.down_sample_layer = None
        if self.downsample:
            self.downsample = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
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
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        #self.conv1 = _conv3x3(3, 16, stride=1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        self.bn1 = _bn(16)
        #self.bn2 = _bn1d(num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.relu = ops.ReLU()

        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))

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

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.fc = _fc(out_channels[2], num_classes)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, has_bias=False)
        self.avgpool = nn.AvgPool2d(8, stride=1)

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

        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.shape[0], -1)


def resnet20(class_num=10):
    """
    Get ResNet20 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet18 neural network.

    Examples:
        >>> net = resnet20(10)
    """
    return ResNet(BasicBlock,
                  [3, 3, 3],
                  [16, 16, 32],
                  [16, 32, 64],
                  [1, 2, 2],
                  class_num)
