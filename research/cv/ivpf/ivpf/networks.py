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
Network blocks.
"""
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype

import mindspore.nn as nn
from mindspore.ops import operations as ops

UNIT_TESTING = True


class Swish(nn.Cell):
    """Swish activation."""
    def __init__(self, train_beta=True):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(
                Tensor(
                    1.0,
                    mstype.float32),
                name='w',
                requires_grad=True)
        else:
            self.weight = 1.0

        self.mul = ops.Mul()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        """construct"""
        return self.mul(x, self.sigmoid(self.mul(self.weight, x)))


class Conv2dSwish(nn.Cell):
    """Conv2d-Norm-Act operation."""
    def __init__(self, n_inputs, n_outputs, kernel_size=3,
                 stride=1, padding=0, bias=True):
        super(Conv2dSwish, self).__init__()
        if n_outputs % 3 == 0:
            num_groups = 3
        elif n_outputs % 2 == 0:
            num_groups = 2
        else:
            num_groups = 1

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=n_outputs)
        self.swish = Swish(train_beta=True)
        self.nn = nn.Conv2d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            has_bias=bias)

    def construct(self, x):
        """construct"""
        x = self.nn(x)
        x = self.group_norm(x)
        x = self.swish(x)
        return x


class DenseLayer(nn.Cell):
    """Dense layer."""
    def __init__(self, n_inputs, growth, kernel, Conv2dAct):
        super(DenseLayer, self).__init__()

        conv1x1 = Conv2dAct(
            n_inputs, n_inputs, kernel_size=1, stride=1,
            padding=0, bias=True)

        self.nn = nn.SequentialCell(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, kernel_size=kernel, stride=1,
                padding=1, bias=True),
        )

        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        """construct"""
        h = self.nn(x)
        h = self.cat((x, h))
        return h


class DenseBlock(nn.Cell):
    """Dense Block."""
    def __init__(self, n_inputs, n_outputs, depth, kernel, Conv2dAct):
        super(DenseBlock, self).__init__()

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(n_inputs, growth, kernel, Conv2dAct))
            n_inputs += growth
            future_growth -= growth

        self.nn = nn.SequentialCell(*layers)

    def construct(self, x):
        """construct"""
        return self.nn(x)


class Identity(nn.Cell):
    """Identity layer."""
    def construct(self, x):
        """construct"""
        return x


class NN(nn.Cell):
    """Dense Block."""
    def __init__(self, n_channels, c_in, c_out, depth, kernel=3):
        super(NN, self).__init__()

        Conv2dAct = Conv2dSwish

        layers = [
            DenseBlock(
                n_inputs=c_in,
                n_outputs=n_channels + c_in,
                depth=depth,
                kernel=kernel,
                Conv2dAct=Conv2dAct)]

        layers += [
            nn.Conv2d(
                n_channels + c_in,
                c_out,
                kernel,
                padding=1,
                pad_mode='pad',
                has_bias=True)
        ]

        self.nn = nn.SequentialCell(*layers)

        if not UNIT_TESTING:
            self.nn[-1].weight.set_data(ops.ZerosLike()(self.nn[-1].weight))
            self.nn[-1].bias.set_data(ops.ZerosLike()(self.nn[-1].bias))

    def construct(self, x):
        """construct"""
        return self.nn(x)
