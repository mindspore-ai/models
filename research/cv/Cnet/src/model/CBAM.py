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

import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
import mindspore.ops as ops
from mindspore import Tensor


class Base_conv(nn.Cell):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 relu=True,
                 bn=True,
                 bias=False):
        super(Base_conv, self).__init__()
        self.out_channels = out_channel
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode='pad',
                              padding=padding,
                              has_bias=bias,
                              weight_init=weight_variable())
        self.bn = nn.BatchNorm2d(
            num_features=out_channel, momentum=0.99, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, feature):
        feature = self.conv(feature)
        if self.bn is not None:
            feature = self.bn(feature)
        if self.relu is not None:
            feature = self.relu(feature)
        return feature


def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)


class Flatten(nn.Cell):

    def construct(self, x):
        return x.view(x.shape[0], -1)


class ChannelGate(nn.Cell):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.SequentialCell(
            Flatten(),
            nn.Dense(in_channels=gate_channels,
                     out_channels=gate_channels // reduction_ratio,
                     weight_init=weight_variable()),
            nn.ReLU(),
            nn.Dense(in_channels=gate_channels // reduction_ratio,
                     out_channels=gate_channels,
                     weight_init=weight_variable()),
        )

        self.pools = ['avg', 'max']
        self.avg_pool = ops.ReduceMean(keep_dims=True)

        self.act = nn.Sigmoid()

    def construct(self, features):
        n, c, h, _ = features.shape

        channel_att_sum = 0
        if 'avg' in self.pools:
            avg_pool = ops.ReduceMean(keep_dims=True)(features, (2, 3))
            channel_att_sum += self.mlp(avg_pool)

        if 'max' in self.pools:
            if h == 32:
                max_pool = nn.MaxPool2d(8, 8)(features)
                max_pool = nn.MaxPool2d(4, 4)(max_pool)
            elif h == 16:
                max_pool = nn.MaxPool2d(8, 8)(features)
                max_pool = nn.MaxPool2d(2, 2)(max_pool)
            else:
                max_pool = nn.MaxPool2d(h, h)(features)
            channel_att_sum += self.mlp(max_pool)

        scales = self.act(channel_att_sum).view((n, c, 1, 1))

        return features * scales


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(Tensor.shape[0], Tensor.shape[1], -1)
    s, _ = tensor_flatten.max(axis=2, keepdim=True)
    outputs = s + ops.Log()(ops.Exp()(tensor_flatten - s).sum(dim=2,
                                                              keepdim=True))
    return outputs


class ChannelPool(nn.Cell):

    def construct(self, x):
        return ops.Concat(1)(
            (ops.ExpandDims()(x.max(axis=1),
                              1), ops.ExpandDims()(x.mean(axis=1), 1)))


class SpatialGate(nn.Cell):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.sigmoid = nn.Sigmoid()
        self.spatial = Base_conv(2,
                                 1,
                                 kernel_size,
                                 stride=1,
                                 padding=(kernel_size - 1) // 2,
                                 relu=False)

    def construct(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scales = self.sigmoid(x_out)
        return x * scales


class CBAM(nn.Cell):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def construct(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
