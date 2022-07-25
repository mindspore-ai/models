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
"""LeNet5_SNN."""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from src.ifnode import IFNode
import numpy as np


def init_weight(inC, outC, kernel):
    key = 1 / (inC * kernel * kernel)
    weight = np.random.uniform(-key**0.5, key**0.5, (outC, inC, kernel, kernel)).astype(np.float32)
    return Tensor(weight)


def init_bias(inC, outC, kernel):
    key = 1 / (inC * kernel * kernel)
    weight = np.random.uniform(-key**0.5, key**0.5, (outC)).astype(np.float32)
    return Tensor(weight)


def init_dense_weight(inC, outC):
    key = 1 / inC
    weight = np.random.uniform(-key ** 0.5, key ** 0.5, (outC, inC)).astype(np.float32)
    return Tensor(weight)


def init_dense_bias(inC, outC):
    key = 1 / inC
    weight = np.random.uniform(-key ** 0.5, key ** 0.5, (outC)).astype(np.float32)
    return Tensor(weight)


class Conv2d_Block(nn.Cell):
    """
    block: conv2d + ifnode
    """
    def __init__(self, in_channels, out_channels, weight_init, bias_init, kernel_size=3, stride=1,
                 pad_mode='pad', padding=1, has_bias=True):
        super(Conv2d_Block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, pad_mode=pad_mode, padding=padding, has_bias=has_bias,
                                weight_init=weight_init, bias_init=bias_init)
        self.ifnode = IFNode()

    def construct(self, x_in):
        x, v1 = x_in
        out = self.conv2d(x)
        out, v1 = self.ifnode(out, v1)
        return (out, v1)


class Dense_Block(nn.Cell):
    """
    block: dense + ifnode
    """
    def __init__(self, in_channels, out_channels, weight_init, bias_init):
        super(Dense_Block, self).__init__()
        self.dense = nn.Dense(in_channels=in_channels, out_channels=out_channels,
                              weight_init=weight_init, bias_init=bias_init)
        self.ifnode = IFNode()

    def construct(self, x_in):
        x, v1 = x_in
        out = self.dense(x)
        out, v1 = self.ifnode(out, v1)
        return out, v1

class snn_lenet(nn.Cell):
    """
    snn backbone for lenet with graph mode
    """
    def __init__(self, num_class=10, num_channel=3):
        super(snn_lenet, self).__init__()
        self.T = 100
        self.conv1 = Conv2d_Block(in_channels=num_channel, out_channels=16,
                                  weight_init=init_weight(num_channel, 16, 3), bias_init=init_bias(num_channel, 16, 3))

        self.conv2 = Conv2d_Block(in_channels=16, out_channels=16, stride=2,
                                  weight_init=init_weight(16, 16, 3), bias_init=init_bias(16, 16, 3))

        self.conv3 = Conv2d_Block(in_channels=16, out_channels=32,
                                  weight_init=init_weight(16, 32, 3), bias_init=init_bias(16, 32, 3))

        self.conv4 = Conv2d_Block(in_channels=32, out_channels=32, stride=2,
                                  weight_init=init_weight(32, 32, 3), bias_init=init_bias(32, 32, 3))

        self.conv5 = Conv2d_Block(in_channels=32, out_channels=64,
                                  weight_init=init_weight(32, 64, 3), bias_init=init_bias(32, 64, 3))

        self.conv6 = Conv2d_Block(in_channels=64, out_channels=64, stride=2,
                                  weight_init=init_weight(64, 64, 3), bias_init=init_bias(64, 64, 3))

        self.dense1 = Dense_Block(in_channels=64 * 4 * 4, out_channels=32,
                                  weight_init=init_dense_weight(64 * 4 * 4, 32),
                                  bias_init=init_dense_bias(64 * 4 * 4, 32))

        self.fc = nn.Dense(32, num_class, weight_init=init_dense_weight(32, num_class),
                           bias_init=init_dense_bias(32, num_class))
        self.end_ifnode = IFNode(fire=False)

    def construct(self, x_in):
        """forward the snn-lenet block"""
        x = x_in
        v1 = v2 = v3 = v4 = v5 = v6 = v7 = v8 = 0.0
        for _ in range(self.T):
            x, v1 = self.conv1((x_in, v1))
            x, v2 = self.conv2((x, v2))
            x, v3 = self.conv3((x, v3))
            x, v4 = self.conv4((x, v4))
            x, v5 = self.conv5((x, v5))
            x, v6 = self.conv6((x, v6))
            x = P.Reshape()(x, (-1, 64 * 4 * 4))
            x, v7 = self.dense1((x, v7))
            x = self.fc(x)
            x, v8 = self.end_ifnode(x, v8)
        return x / self.T
