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
"""Res2Net."""
import math
import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from src.model_utils.config import config


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    """conv_variance_scaling_initializer"""
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1.0, fan_in)
    stddev = (scale ** 0.5) / 0.87962566103423978
    if config.net_name == "res2net152":
        stddev = scale ** 0.5
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(
        out_channel * in_channel * kernel_size * kernel_size
    )
    weight = np.reshape(
        weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    """_weight_variable"""
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    res = 0
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        res = 1
    elif nonlinearity == "tanh":
        res = 5.0 / 3
    elif nonlinearity == "relu":
        res = math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            neg_slope = 0.01
        elif (
                not isinstance(param, bool)
                and isinstance(param, int)
                or isinstance(param, float)
        ):
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
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
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
    """_calculate_correct_fan"""
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Unsupported mode {}, please use one of {}".format(
                mode, valid_modes)
        )
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_normal(inputs_shape, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    """kaiming_normal"""
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0.0, mode="fan_in", nonlinearity="leaky_relu"):
    """kaiming_uniform"""
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv3x3"""
    if use_se:
        weight = conv_variance_scaling_initializer(
            in_channel, out_channel, kernel_size=3
        )
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = Tensor(
            kaiming_normal(weight_shape, mode="fan_out", nonlinearity="relu")
        )
        if config.net_name == "res2net152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
            weight_init=weight,
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=0,
        pad_mode="same",
        weight_init=weight,
    )


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv1x1"""
    if use_se:
        weight = conv_variance_scaling_initializer(
            in_channel, out_channel, kernel_size=1
        )
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = Tensor(
            kaiming_normal(weight_shape, mode="fan_out", nonlinearity="relu")
        )
        if config.net_name == "res2net152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=stride,
            padding=0,
            pad_mode="pad",
            weight_init=weight,
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=stride,
        padding=0,
        pad_mode="same",
        weight_init=weight,
    )


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv7x7"""
    if use_se:
        weight = conv_variance_scaling_initializer(
            in_channel, out_channel, kernel_size=7
        )
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = Tensor(
            kaiming_normal(weight_shape, mode="fan_out", nonlinearity="relu")
        )
        if config.net_name == "res2net152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=7,
            stride=stride,
            padding=3,
            pad_mode="pad",
            weight_init=weight,
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=7,
        stride=stride,
        padding=0,
        pad_mode="same",
        weight_init=weight,
    )


def _bn(channel, res_base=False):
    """_bn"""
    if res_base:
        return nn.BatchNorm2d(
            channel,
            eps=1e-5,
            momentum=0.1,
            gamma_init=1,
            beta_init=0,
            moving_mean_init=0,
            moving_var_init=1,
        )
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
        gamma_init=1,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _bn_last(channel):
    """_bn_last"""
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
        gamma_init=0,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _fc(in_channel, out_channel, use_se=False):
    """_fc"""
    if use_se:
        weight = np.random.normal(
            loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(
            np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32
        )
    else:
        weight_shape = (out_channel, in_channel)
        weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
        if config.net_name == "res2net152":
            weight = _weight_variable(weight_shape)
    return nn.Dense(
        in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0
    )


class Residual2Block(nn.Cell):
    """
    Res2Net residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-Res2Net50 net. Default: False.
        se_block(bool): Use se block in SE-Res2Net50 net. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """

    expansion = 4

    def __init__(
            self,
            in_channel,
            out_channel,
            stride=1,
            use_se=False,
            se_block=False,
            baseWidth=26,
            scale=4,
            stype="normal",
    ):
        super(Residual2Block, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block

        assert scale > 1, "Res2Net is ResNet when scale = 1"
        width = int(math.floor(out_channel //
                               self.expansion * (baseWidth / 64.0)))
        channel = width * scale

        self.conv1 = _conv1x1(in_channel, channel,
                              stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)

        if stype == "stage":
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=stride, pad_mode="same")

        self.convs = nn.CellList()
        self.bns = nn.CellList()
        for _ in range(scale - 1):
            self.convs.append(
                _conv3x3(width, width, stride=stride, use_se=self.use_se))
            self.bns.append(_bn(width))

        self.conv3 = _conv1x1(channel, out_channel,
                              stride=1, use_se=self.use_se)
        self.bn3 = _bn(out_channel)
        if config.optimizer == "Thor" or config.net_name == "res2net152":
            self.bn3 = _bn_last(out_channel)
        if self.se_block:
            self.se_global_pool = P.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(
                out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4),
                                  out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = P.Mul()
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            if stride == 1:
                self.down_sample_layer = nn.SequentialCell(
                    [
                        _conv1x1(in_channel, out_channel,
                                 stride, use_se=self.use_se),
                        _bn(out_channel),
                    ]
                )
            else:
                self.down_sample_layer = nn.SequentialCell(
                    [
                        nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
                        _conv1x1(in_channel, out_channel,
                                 1, use_se=self.use_se),
                        _bn(out_channel),
                    ]
                )
        self.add = P.Add()
        self.inplace_add = P.InplaceAdd((0, 1, 2, 3))
        self.scale = scale
        self.width = width
        self.stride = stride
        self.stype = stype
        self.split = P.Split(axis=1, output_num=scale)
        self.cat = P.Concat(axis=1)

    def construct(self, x):
        """construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = self.split(out)

        sp = self.convs[0](spx[0])
        sp = self.relu(self.bns[0](sp))
        out = sp
        for i in range(1, self.scale - 1):
            if self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp[:, :, :, :]  # to avoid bug in mindspore
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = self.cat((out, sp))

        if self.stype == "normal":
            out = self.cat((out, spx[self.scale - 1]))
        elif self.stype == "stage":
            out = self.cat((out, self.pool(spx[self.scale - 1])))

        out = self.conv3(out)
        out = self.bn3(out)
        if self.se_block:
            out_se = out
            out = self.se_global_pool(out, (2, 3))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = F.reshape(out, F.shape(out) + (1, 1))
            out = self.se_mul(out, out_se)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


class Res2Net(nn.Cell):
    """
    Res2Net architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
        use_se (bool): Enable SE-Res2Net50 net. Default: False.
        se_block(bool): Use se block in SE-Res2Net50 net in layer 3 and layer 4. Default: False.
        res_base (bool): Enable parameter setting of res2net18. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> Res2Net(Residual2Block,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(
            self,
            block,
            layer_nums,
            in_channels,
            out_channels,
            strides,
            num_classes,
            use_se=False,
            res_base=False,
    ):
        super(Res2Net, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError(
                "the length of layer_num, in_channels, out_channels list must be 4!"
            )
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
        self.bn1_0 = _bn(32)
        self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
        self.bn1_1 = _bn(32)
        self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)

        self.bn1 = _bn(64, self.res_base)
        self.relu = P.ReLU()

        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(
            block,
            layer_nums[0],
            in_channel=in_channels[0],
            out_channel=out_channels[0],
            stride=strides[0],
            use_se=self.use_se,
        )
        self.layer2 = self._make_layer(
            block,
            layer_nums[1],
            in_channel=in_channels[1],
            out_channel=out_channels[1],
            stride=strides[1],
            use_se=self.use_se,
        )
        self.layer3 = self._make_layer(
            block,
            layer_nums[2],
            in_channel=in_channels[2],
            out_channel=out_channels[2],
            stride=strides[2],
            use_se=self.use_se,
            se_block=self.se_block,
        )
        self.layer4 = self._make_layer(
            block,
            layer_nums[3],
            in_channel=in_channels[3],
            out_channel=out_channels[3],
            stride=strides[3],
            use_se=self.use_se,
            se_block=self.se_block,
        )

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=self.use_se)

    def _make_layer(
            self,
            block,
            layer_num,
            in_channel,
            out_channel,
            stride,
            use_se=False,
            se_block=False,
    ):
        """
        Make stage network of Res2Net.

        Args:
            block (Cell): Res2net block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-Res2Net50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        res2net_block = block(
            in_channel, out_channel, stride=stride, use_se=use_se, stype="stage"
        )
        layers.append(res2net_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                res2net_block = block(
                    out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(res2net_block)
            res2net_block = block(
                out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block
            )
            layers.append(res2net_block)
        else:
            for _ in range(1, layer_num):
                res2net_block = block(
                    out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(res2net_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        x = self.conv1_0(x)
        x = self.bn1_0(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)

        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def res2net50(class_num=10):
    """
    Get Res2Net50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of Res2Net50 neural network.

    Examples:
        >>> net = res2net50(10)
    """
    return Res2Net(
        Residual2Block,
        [3, 4, 6, 3],
        [64, 256, 512, 1024],
        [256, 512, 1024, 2048],
        [1, 2, 2, 2],
        class_num,
    )


def se_res2net50(class_num=1001):
    """
    Get SE-Res2Net50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of SE-Res2Net50 neural network.

    Examples:
        >>> net = se-res2net50(1001)
    """
    return Res2Net(
        Residual2Block,
        [3, 4, 6, 3],
        [64, 256, 512, 1024],
        [256, 512, 1024, 2048],
        [1, 2, 2, 2],
        class_num,
        use_se=True,
    )


def res2net101(class_num=1001):
    """
    Get Res2Net101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of Res2Net101 neural network.

    Examples:
        >>> net = res2net101(1001)
    """
    return Res2Net(
        Residual2Block,
        [3, 4, 23, 3],
        [64, 256, 512, 1024],
        [256, 512, 1024, 2048],
        [1, 2, 2, 2],
        class_num,
    )


def res2net152(class_num=1001):
    """
    Get Res2Net152 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of Res2Net152 neural network.

    Examples:
        # >>> net = res2net152(1001)
    """
    return Res2Net(
        Residual2Block,
        [3, 8, 36, 3],
        [64, 256, 512, 1024],
        [256, 512, 1024, 2048],
        [1, 2, 2, 2],
        class_num,
    )
