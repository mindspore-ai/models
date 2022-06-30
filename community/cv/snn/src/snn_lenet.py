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
from src.ifnode import IFNode_GRAPH, IFNode_PYNATIVE
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


class snn_lenet_graph(nn.Cell):
    """
    snn backbone for lenet with graph mode
    """
    def __init__(self, num_class=10, num_channel=3):
        super(snn_lenet_graph, self).__init__()
        self.T = 100
        self.conv1 = nn.Conv2d(num_channel, 16, 3, stride=1, pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init_weight(num_channel, 16, 3), bias_init=init_bias(num_channel, 16, 3))
        self.ifnode1 = IFNode_GRAPH()
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init_weight(16, 16, 3), bias_init=init_bias(16, 16, 3))
        self.ifnode2 = IFNode_GRAPH()
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init_weight(16, 32, 3), bias_init=init_bias(16, 32, 3))
        self.ifnode3 = IFNode_GRAPH()
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init_weight(32, 32, 3), bias_init=init_bias(32, 32, 3))
        self.ifnode4 = IFNode_GRAPH()
        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init_weight(32, 64, 3), bias_init=init_bias(32, 64, 3))
        self.ifnode5 = IFNode_GRAPH()
        self.conv6 = nn.Conv2d(64, 64, 3, stride=2, pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init_weight(64, 64, 3), bias_init=init_bias(64, 64, 3))
        self.ifnode6 = IFNode_GRAPH()
        self.fc1 = nn.Dense(64 * 4 * 4, 32, weight_init=init_dense_weight(64 * 4 * 4, 32),
                            bias_init=init_dense_bias(64 * 4 * 4, 32))
        self.ifnode7 = IFNode_GRAPH()
        self.fc2 = nn.Dense(32, num_class, weight_init=init_dense_weight(32, num_class),
                            bias_init=init_dense_bias(32, num_class))
        self.ifnode8 = IFNode_GRAPH(fire=False)

    def construct(self, x_in):
        """forward the snn-lenet block"""
        x = x_in
        v1 = v2 = v3 = v4 = v5 = v6 = v7 = v8 = 0.0
        for _ in range(self.T):
            x = self.conv1(x_in)
            x, v1 = self.ifnode1(x, v1)
            x = self.conv2(x)
            x, v2 = self.ifnode2(x, v2)
            x = self.conv3(x)
            x, v3 = self.ifnode3(x, v3)
            x = self.conv4(x)
            x, v4 = self.ifnode4(x, v4)
            x = self.conv5(x)
            x, v5 = self.ifnode5(x, v5)
            x = self.conv6(x)
            x, v6 = self.ifnode6(x, v6)
            x = P.Reshape()(x, (-1, 64 * 4 * 4))
            x = self.fc1(x)
            x, v7 = self.ifnode7(x, v7)
            x = self.fc2(x)
            x, v8 = self.ifnode8(x, v8)
        return x / self.T


class snn_lenet_pynative(nn.Cell):
    """
    snn backbone for lenet with pynative mode
    """
    def __init__(self, num_class=10, num_channel=3):
        super(snn_lenet_pynative, self).__init__()
        self.T = 100
        self.conv1 = nn.SequentialCell([nn.Conv2d(num_channel, 16, 3, stride=1, pad_mode='pad', padding=1,
                                                  has_bias=True, weight_init=init_weight(num_channel, 16, 3),
                                                  bias_init=init_bias(num_channel, 16, 3)),
                                        IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.conv2 = nn.SequentialCell([nn.Conv2d(16, 16, 3, stride=2, pad_mode='pad', padding=1, has_bias=True,
                                                  weight_init=init_weight(16, 16, 3), bias_init=init_bias(16, 16, 3)),
                                        IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.conv3 = nn.SequentialCell([nn.Conv2d(16, 32, 3, stride=1, pad_mode='pad', padding=1, has_bias=True,
                                                  weight_init=init_weight(16, 32, 3), bias_init=init_bias(16, 32, 3)),
                                        IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.conv4 = nn.SequentialCell([nn.Conv2d(32, 32, 3, stride=2, pad_mode='pad', padding=1, has_bias=True,
                                                  weight_init=init_weight(32, 32, 3), bias_init=init_bias(32, 32, 3)),
                                        IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.conv5 = nn.SequentialCell([nn.Conv2d(32, 64, 3, stride=1, pad_mode='pad', padding=1, has_bias=True,
                                                  weight_init=init_weight(32, 64, 3), bias_init=init_bias(32, 64, 3)),
                                        IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.conv6 = nn.SequentialCell([nn.Conv2d(64, 64, 3, stride=2, pad_mode='pad', padding=1, has_bias=True,
                                                  weight_init=init_weight(64, 64, 3), bias_init=init_bias(64, 64, 3)),
                                        IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.fc1 = nn.SequentialCell([nn.Dense(64 * 4 * 4, 32,
                                               weight_init=init_dense_weight(64 * 4 * 4, 32),
                                               bias_init=init_dense_bias(64 * 4 * 4, 32)),
                                      IFNode_PYNATIVE(v_threshold=1.0, v_reset=None)])

        self.fc2 = nn.Dense(32, num_class, weight_init=init_dense_weight(32, num_class),
                            bias_init=init_dense_bias(32, num_class))

        self.outlayer = IFNode_PYNATIVE(v_threshold=1.0, v_reset=None, fire=False)

    def construct(self, x_in):
        """forward the snn-lenet block"""
        x = x_in
        for _ in range(self.T):
            x = self.conv1(x_in)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = P.Reshape()(x, (-1, 64 * 4 * 4))
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.outlayer(x)
        return x / self.T

    def reset_net(self):
        """each batch should reset the accumulated value of the net such as self.v"""
        for item in self.cells():
            if isinstance(type(item), type(nn.SequentialCell())):
                if hasattr(item[-1], 'reset'):
                    item[-1].reset()
            else:
                if hasattr(item, 'reset'):
                    item.reset()
