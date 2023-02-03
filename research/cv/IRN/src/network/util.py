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
"""utility functions for network constructing"""

import mindspore
import mindspore.nn as nn
import mindspore.common.initializer as init
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype, context

class ReconstructionLoss(nn.Cell):
    """L1 and L2 Loss """
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32

        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.abs = ops.Abs()
        self.cast = ops.Cast()
        self.sqrt = ops.Sqrt()
        self.eps = Tensor(eps, self.cast_type)

    def construct(self, x, target):
        '''construct method for loss'''
        x = self.cast(x, self.cast_type)
        target = self.cast(target, self.cast_type)
        if self.losstype == 'l2':
            return self.mean(self.sum((x - target)**2, (1, 2, 3)))
        if self.losstype == 'l1':
            diff = x - target
            return self.mean(self.sum(self.sqrt(diff * diff + self.eps), (1, 2, 3)))
        print("reconstruction loss type error!")
        return 0

def initialize_weights(net_l, scale=1):
    """weights initialization"""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for _, m in net.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(
                    init.initializer(init.HeNormal(negative_slope=0, mode='fan_in'),
                                     m.weight.shape, m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)  # for residual block
                if m.bias is not None:
                    m.bias.set_data(init.initializer(
                        init.Zero(), m.bias.shape, m.bias.dtype))
                    m.bias.requires_grad = True
            elif isinstance(m, nn.Dense):
                m.weight.set_data(
                    init.initializer(init.HeNormal(negative_slope=0, mode='fan_in'),
                                     m.weight.shape, m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)  # for residual block
                if m.bias is not None:
                    m.bias.set_data(init.initializer(
                        init.Zero(), m.bias.shape, m.bias.dtype))
                    m.bias.requires_grad = True
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.set_data(init.initializer(
                    init.Constant(1), m.weight.shape, m.weight.dtype))
                m.bias.set_data(init.initializer(
                    init.Zero(), m.bias.shape, m.bias.dtype))
                m.bias.requires_grad = True


def initialize_weights_xavier(net_l, scale=1):
    """xavier initialization for weight"""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for _, m in net.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(init.initializer(
                    init.XavierUniform(), m.weight.shape, m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)  # for residual block
                if m.bias is not None:
                    m.bias.set_data(init.initializer(
                        init.Zero(), m.bias.shape, m.bias.dtype))
                    m.bias.requires_grad = True
            elif isinstance(m, nn.Dense):
                m.weight.set_data(init.initializer(
                    init.XavierUniform(), m.weight.shape, m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)
                if m.bias is not None:
                    m.bias.set_data(init.initializer(
                        init.Zero(), m.bias.shape, m.bias.dtype))
                    m.bias.requires_grad = True
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.set_data(init.initializer(
                    init.Constant(1), m.weight.shape, m.weight.dtype))
                m.bias.set_data(init.initializer(
                    init.Zero(), m.bias.shape, m.bias.dtype))
                m.bias.requires_grad = True


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.SequentialCell(layers)


class ResidualBlock_noBN(nn.Cell):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, "pad", 1, has_bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, "pad", 1, has_bias=True)

        self.relu = nn.Relu()

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class GroupConv(nn.Cell):
    """
    group convolution operation.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.

    Returns:
        tensor, output tensor.

    Examples:
        https://gitee.com/mindspore/models/blob/master/official/cv/ResNeXt/src/backbone/resnet.py
    """

    def __init__(self, out_channels, kernel_size, stride=1, pad_mode="pad", pad=0, groups=1):
        super(GroupConv, self).__init__()
        assert out_channels % groups == 0
        self.groups = groups
        self.convs = []
        self.op_split = ops.Split(axis=1, output_num=self.groups)
        self.op_split_w = ops.Split(axis=0, output_num=self.groups)
        self.op_concat = ops.Concat(axis=1)
        self.cast = ops.Cast()
        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32
        for _ in range(groups):
            self.convs.append(mindspore.ops.Conv2D(out_channels//groups,
                                                   kernel_size=kernel_size, stride=stride,
                                                   pad=pad, pad_mode=pad_mode, group=1))

    def construct(self, x, weight):
        features = self.op_split(x)
        weights = self.op_split_w(weight)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + \
                (self.convs[i](
                    self.cast(features[i], self.cast_type), weights[i]),)
        out = self.op_concat(outputs)
        return out



class GroupTransConv(nn.Cell):
    """
    group transposed convolution operation.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.


    Returns:
        tensor, output tensor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            pad_mode="pad", pad=0, groups=1, has_bias=False, weight_init='normal'):
        super(GroupTransConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        self.convsTrans = []
        self.op_split = ops.Split(axis=1, output_num=self.groups)
        self.op_split_w = ops.Split(axis=0, output_num=self.groups)

        self.op_concat = ops.Concat(axis=1)
        self.cast = ops.Cast()
        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32
        weights = self.op_split_w(weight_init)

        for i in range(groups):
            self.convsTrans.append(nn.Conv2dTranspose(in_channels//groups, out_channels//groups,
                                                      kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                                      padding=pad, pad_mode=pad_mode, group=1, weight_init=weights[i]))
            self.convsTrans[i].weight.requires_grad = False

    def construct(self, x):
        features = self.op_split(x)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + \
                (self.convsTrans[i](self.cast(features[i], self.cast_type)),)
        out = self.op_concat(outputs)
        return out
