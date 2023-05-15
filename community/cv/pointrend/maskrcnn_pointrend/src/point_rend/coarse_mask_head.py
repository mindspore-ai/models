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
'''coarse mask head'''

import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from maskrcnn.model_utils.device_adapter import get_device_id


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''kaiming normal'''
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
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
    '''calculate correct fan'''
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=self.kernel_size, stride=self.stride,
                                pad_mode=self.pad_mode, padding=self.padding, has_bias=True)

    def construct(self, x):
        '''construct'''
        x = self.conv2d(x, self.weight)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CoarseMaskHead(nn.Cell):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, num_classes, input_channel=256, input_height=14, input_width=14):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimension of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super(CoarseMaskHead, self).__init__()

        # fmt: off
        # todo 通道数改为81，无效label为0不变
        self.num_classes = num_classes
        # self.num_classes = num_classes -1
        conv_dim = 256
        self.fc_dim = 1024
        num_fc = 2
        self.output_side_resolution = 7
        self.input_channels = input_channel
        self.input_h = input_height
        self.input_w = input_width
        # fmt: on
        self.relu = ops.ReLU()
        self.flatten = ops.Flatten()

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                in_channels=self.input_channels,
                out_channels=conv_dim,
                kernel_size=1,
                stride=1,
                pad_mode='vaild',
                padding=0,
                activation=ops.ReLU(),
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=2,
                                                 stride=2, pad_mode='same', padding=0, has_bias=True)
        self.conv_layers.append(self.reduce_spatial_dim_conv)
        self.conv_layers.append(nn.ReLU())
        self.conv_seq = nn.SequentialCell(self.conv_layers)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for _ in range(num_fc):
            fc = nn.Dense(input_dim, self.fc_dim)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = self.num_classes * self.output_side_resolution * self.output_side_resolution

        self.fcs_seq = nn.SequentialCell(self.fcs)

        self.prediction = nn.Dense(self.fc_dim, output_dim)


    def construct(self, x):
        '''construct'''
        # unlike BaseMaskRCNNHead, this head only outputs intermediate
        # features, because the features will be used later by PointHead.
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        x = self.conv_seq(x)
        x = self.flatten(x)
        for layer in self.fcs_seq.cell_list:
            x = self.relu(layer(x))
        x = self.prediction(x)
        x = x.view(N, self.num_classes, self.output_side_resolution, self.output_side_resolution)
        return x


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=get_device_id())
    shape = (128, 256, 14, 14)
    stdnormal = ops.StandardNormal(seed=2)
    xx = stdnormal(shape)
    net = CoarseMaskHead(num_classes=81)
    out = net(xx)
    print(out)
