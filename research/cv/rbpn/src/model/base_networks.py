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

"""basenetwork"""

import mindspore.nn as nn
import mindspore.ops as ops


class DenseBlock(nn.Cell):
    """Structure of DenseBlock network"""

    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()

        self.fc = nn.Dense(input_size, output_size, has_bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(output_size, momentum=0.1)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU(channel=output_size)
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        """dense block unit """
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)
        if self.activation is not None:
            return self.act(out)
        return out


class ConvBlock(nn.Cell):
    """Structure of ConvBlock network"""

    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, pad_mode='pad', padding=padding,
                              has_bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU(channel=output_size)
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        """dense block unit """
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        return out


class DeconvBlock(nn.Cell):
    """Structure of DeconvBlock network"""

    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Conv2dTranspose(input_size, output_size, kernel_size, stride, pad_mode='pad', padding=padding,
                                         has_bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU(channel=output_size)
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        """DeconvBlock unit"""
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        return out


class ResnetBlock(nn.Cell):
    """Structure of ResnetBlock network"""

    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, pad_mode='pad', padding=padding,
                               has_bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, pad_mode='pad', padding=padding,
                               has_bias=bias)
        self.add_ops = ops.Add()
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU(channel=num_filter)
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        """ResnetBlock unit
        Args:
            x(Tensor): image
        Outputs:
            Tensor
        """
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)
        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)
        out = self.add_ops(out, residual)

        if self.activation is not None:
            out = self.act(out)
        return out


class UpBlock(nn.Cell):
    """Structure of UpBlock network"""

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                  activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)

    def construct(self, x):
        """UpBlock compute loss"""
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DUpBlock(nn.Cell):
    """Structure of D_UpBlock network"""

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
                 num_stages=1, bias=True, activation='relu', norm=None):
        super(DUpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                  activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)

    def construct(self, x):
        """DUpBlock compute dense loss"""
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Cell):
    """Structure of DownBlock network"""

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                      activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)

    def construct(self, x):
        """down block unit"""
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DDownBlock(nn.Cell):
    """Structure of D_DownBlock network"""

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(DDownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                      activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias,
                                    activation=activation, norm=None)

    def construct(self, x):
        """DDownBlock compute dense loss"""
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0
