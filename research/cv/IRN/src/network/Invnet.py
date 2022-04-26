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
"""Invertible Rescaling Network construction"""

import math
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as ops
from mindspore import dtype as mstype, context
import src.network.util as mutil


class DenseBlock(nn.Cell):
    """
        define DenseBlock
    """

    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 'pad', 1, has_bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3,
                               1, 'pad', 1, has_bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc,
                               3, 1, 'pad', 1, has_bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc,
                               3, 1, 'pad', 1, has_bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc,
                               channel_out, 3, 1, 'pad', 1, has_bias=bias)
        self.lrelu = nn.LeakyReLU(0.2)
        self.cat = ops.Concat(1)

        if init == 'xavier':
            mutil.initialize_weights_xavier(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def construct(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.cat((x, x1))))
        x3 = self.lrelu(self.conv3(self.cat((x, x1, x2))))
        x4 = self.lrelu(self.conv4(self.cat((x, x1, x2, x3))))
        x5 = self.conv5(self.cat((x, x1, x2, x3, x4)))

        return x5


class InvBlockExp(nn.Cell):
    """
        define Invertible Block
    """

    def __init__(self, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp

        self.F = DenseBlock(self.split_len2, self.split_len1)
        self.G = DenseBlock(self.split_len1, self.split_len2)
        self.H = DenseBlock(self.split_len1, self.split_len2)

        self.sigmoid = ops.Sigmoid()
        self.exp = ops.Exp()
        self.cat = ops.Concat(axis=1)
        self.sum = ops.ReduceSum()
        self.mul = ops.Mul()
        self.div = ops.Div()
        self.rev = False

    def construct(self, x, rev=False):
        '''Construct method for Invertible Block'''
        x1, x2 = x[:, 0:0+self.split_len1], x[:,
                                              self.split_len1:self.split_len1+self.split_len2]

        if not rev:
            y1 = x1 + self.F(x2)
            s = self.clamp * (self.sigmoid(self.H(y1)) * 2 - 1)
            y2 = self.mul(x2, self.exp(s)) + self.G(y1)
        else:
            s = self.clamp * (self.sigmoid(self.H(x1)) * 2 - 1)
            y2 = self.div((x2 - self.G(x1)), self.exp(s))
            y1 = x1 - self.F(y2)

        return self.cat((y1, y2))


class HaarDownsampling(nn.Cell):
    """
        define HaarDownsampling
    """

    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32

        self.channel_in = channel_in

        self.haar_weights = np.ones((4, 1, 2, 2))

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = np.concatenate(
            [self.haar_weights] * self.channel_in, 0)
        self.haar_weights = ms.Tensor(self.haar_weights).astype(self.cast_type)
        self.haar_weights.requires_grad = False

        self.conv2d = mutil.GroupConv(
            out_channels=self.haar_weights.shape[0],
            kernel_size=(
                self.haar_weights.shape[2], self.haar_weights.shape[3]),
            stride=2,
            groups=self.channel_in)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

        self.conv2d_transpose = mutil.GroupTransConv(
            in_channels=self.haar_weights.shape[0],
            out_channels=self.haar_weights.shape[1]*self.channel_in,
            kernel_size=(
                self.haar_weights.shape[2], self.haar_weights.shape[3]),
            stride=2,
            groups=self.channel_in,
            weight_init=self.haar_weights)

    def construct(self, x, rev=False):
        '''Construct method for HaarDownsampling Block'''
        if not rev:
            out = self.conv2d(x, self.haar_weights) / 4.0
            out = self.reshape(
                out, (x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2))
            out = self.transpose(out, (0, 2, 1, 3, 4))
            out = self.reshape(
                out, (x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2))
            return out

        out = self.reshape(
            x, (x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]))
        out = self.transpose(out, (0, 2, 1, 3, 4))
        out = self.reshape(
            out, (x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]))
        return self.conv2d_transpose(out)


class InvRescaleNet(nn.Cell):
    """
        define Invertible Rescaling Network
    """

    def __init__(self, channel_in=3, channel_out=3, block_num=None, down_num=2):
        super(InvRescaleNet, self).__init__()

        if block_num is None:
            block_num = []

        operations = []
        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for _ in range(block_num[i]):
                b = InvBlockExp(current_channel, channel_out)
                operations.append(b)
        self.operations = nn.CellList(operations)

    def construct(self, x, rev=False):
        out = x
        if not rev:
            for ind in range(len(self.operations)):
                out = self.operations[ind](out, rev)
        else:
            for ind in range(len(self.operations)):
                out = self.operations[len(self.operations)-1-ind](out, rev)
        return out


def define_G(opt):
    """
        define Generator network
    """
    opt_net = opt['network_G']
    down_num = int(math.log(opt_net['scale'], 2))

    netG = InvRescaleNet(
        opt_net['in_nc'], opt_net['out_nc'], opt_net['block_num'], down_num)

    return netG
