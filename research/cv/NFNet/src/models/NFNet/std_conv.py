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
""" Convolution with Weight Standardization (StdConv and ScaledStdConv)

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code: https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692

Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Hacked together by / copyright Ross Wightman, 2021.
"""
import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from src.args import args


class ScaledStdConv2dUnit(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same",
                 dilation=1, group=1, has_bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, pad_mode=pad_mode,
                         dilation=dilation,
                         group=group, has_bias=has_bias)
        self.gain = Parameter(Tensor(np.full((self.out_channels, 1, 1, 1), gain_init), mstype.float32))
        self.fan_in = Tensor(np.prod(self.weight[0].shape), mstype.float32)  # gamma * 1 / sqrt(fan-in)
        self.gamma = Tensor(gamma, mstype.float32)
        self.eps = eps

    def construct(self, x):
        """ScaledStdConv2dUnit Construct"""

        mean = ops.ReduceMean(True)(self.weight, (1, 2, 3))
        var = ops.ReduceMean(True)(ops.Square()(self.weight - mean), (1, 2, 3))
        scale = ops.Rsqrt()(ops.Maximum()(var * self.fan_in, self.eps))
        weight = (self.weight - mean) * self.gain * scale
        x = self.conv2d(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        return x


if args.device_target == "GPU":
    ScaledStdConv2dSame = ScaledStdConv2dUnit
elif args.device_target == "Ascend":
    class ScaledStdConv2dSame(nn.Cell):
        """
        group convolution operation.

        Args:
            in_channels (int): Input channels of feature map.
            out_channels (int): Output channels of feature map.
            kernel_size (int): Size of convolution kernel.
            stride (int): Stride size for the group convolution layer.

        Returns:
            tensor, output tensor.
        """

        def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same",
                     dilation=1, group=1, has_bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
            super(ScaledStdConv2dSame, self).__init__()
            self.group = group
            if group > 1:
                assert in_channels % group == 0 and out_channels % group == 0
                self.convs = nn.CellList()
                self.op_split = ops.Split(axis=1, output_num=self.group)
                self.op_concat = ops.Concat(axis=1)
                self.cast = ops.Cast()
                for _ in range(group):
                    self.convs.append(ScaledStdConv2dUnit(in_channels // group, out_channels // group,
                                                          kernel_size, stride=stride, pad_mode=pad_mode,
                                                          dilation=dilation, group=1, has_bias=has_bias, gamma=gamma,
                                                          eps=eps, gain_init=gain_init))
            else:
                self.conv = ScaledStdConv2dUnit(in_channels, out_channels, kernel_size, stride=stride,
                                                pad_mode=pad_mode, dilation=dilation, group=1, has_bias=has_bias,
                                                gamma=gamma, eps=eps, gain_init=gain_init)

        def construct(self, x):
            if self.group > 1:
                features = self.op_split(x)
                outputs = ()
                for i in range(self.group):
                    outputs = outputs + (self.convs[i](self.cast(features[i], mstype.float32)),)
                out = self.op_concat(outputs)
            else:
                out = self.conv(x)
            return out
