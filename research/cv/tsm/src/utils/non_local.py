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
"""non_local"""
import mindspore.nn as nn
import mindspore.ops as ops
from src.model.resnet import ResNet

class _NonLocalBlockND(nn.Cell):
    """NonLocalBlockND"""
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.layer.MaxPool2d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.SequentialCell(
                [conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, weight_init="zeros", bias_init="zeros"),
                 bn(self.in_channels)]
            )
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, weight_init="zeros", bias_init="zeros")

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.SequentialCell([self.g, max_pool_layer])
            self.phi = nn.SequentialCell([self.phi, max_pool_layer])

    def construct(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0)
        transpose = ops.Transpose()

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = transpose(g_x, (0, 2, 1))

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = transpose(theta_x, (0, 2, 1))
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = ops.matmul(theta_x, phi_x)
        f_div_C = ops.Softmax(f, dim=-1)

        y = ops.matmul(f_div_C, g_x)
        y = transpose(y, (0, 2, 1))
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class NL3DWrapper(nn.Cell):
    """NL3DWrapper"""
    def __init__(self, block, n_segment):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.nl = NONLocalBlock3D(block.bn3.num_features)
        self.n_segment = n_segment

    def construct(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)
        x = x.transpose(1, 2).view(nt, c, h, w)
        return x


def make_non_local(net, n_segment):
    """make_non_local"""
    if isinstance(net, ResNet):
        net.layer2 = nn.SequentialCell(
            [NL3DWrapper(net.layer2[0], n_segment),
             net.layer2[1],
             NL3DWrapper(net.layer2[2], n_segment),
             net.layer2[3]]
        )
        net.layer3 = nn.SequentialCell(
            [NL3DWrapper(net.layer3[0], n_segment),
             net.layer3[1],
             NL3DWrapper(net.layer3[2], n_segment),
             net.layer3[3],
             NL3DWrapper(net.layer3[4], n_segment),
             net.layer3[5]]
        )
    else:
        raise NotImplementedError
