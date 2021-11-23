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
"""Vnet network"""
import mindspore.nn as nn
import mindspore.ops


def ELUCons(elu, nchannels):
    """activation function"""

    if elu:
        return mindspore.ops.Elu()
    return nn.PReLU(nchannels)


class LUConv(nn.Cell):
    """convolution with activation function and BN"""

    def __init__(self, nchannels, elu):
        super(LUConv, self).__init__()
        self.relu = ELUCons(elu, nchannels)
        self.conv = nn.Conv3d(nchannels, nchannels, kernel_size=(5, 5, 5), pad_mode='pad', padding=2)
        self.bn = nn.BatchNorm3d(nchannels)

    def construct(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


def _make_nConv(nchannels, depth, elu):
    """make convolution layers"""

    layers = []
    if depth == 1:
        return LUConv(nchannels, elu)
    for _ in range(depth):
        layers.append(LUConv(nchannels, elu))
    return nn.SequentialCell(*layers)


class InputTransition(nn.Cell):
    """input transition module"""

    def __init__(self, out_channels, elu):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(1, 16, kernel_size=(5, 5, 5), pad_mode='pad', padding=2)
        self.bn = nn.BatchNorm3d(16)
        self.relu = ELUCons(elu, 16)
        self.cat = mindspore.ops.Concat(axis=1)

    def construct(self, x):
        out = self.bn(self.conv(x))
        x16 = self.cat((x, x, x, x, x, x, x, x,
                        x, x, x, x, x, x, x, x))
        out = self.relu(mindspore.ops.tensor_add(out, x16))
        return out


class DownTransition(nn.Cell):
    """down transition module"""

    def __init__(self, in_channels, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        self.dropout = dropout
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 2, 2), stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        self.dropout_op = mindspore.ops.Dropout3D()
        self.ops = _make_nConv(out_channels, nConvs, elu)

    def construct(self, x):
        down = self.relu1(self.bn(self.down_conv(x)))
        if self.dropout:
            out, _ = self.dropout_op(down)
        else:
            out = down
        out = self.ops(out)
        out = self.relu2(mindspore.ops.tensor_add(out, down))
        return out


class UpTransition(nn.Cell):
    """up transition module"""

    def __init__(self, in_channels, out_channels, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.dropout = dropout
        self.up_conv = nn.Conv3dTranspose(in_channels, out_channels // 2, kernel_size=(2, 2, 2), stride=2)
        self.bn = nn.BatchNorm3d(out_channels // 2)
        self.relu1 = ELUCons(elu, out_channels // 2)
        self.relu2 = ELUCons(elu, out_channels)
        self.dropout_op1 = mindspore.ops.Dropout3D()
        self.dropout_op2 = mindspore.ops.Dropout3D()
        self.ops = _make_nConv(out_channels, nConvs, elu)
        self.cat = mindspore.ops.Concat(axis=1)

    def construct(self, x, skipx):
        """up transition module construct"""

        if self.dropout:
            out, _ = self.dropout_op1(x)
            skipx_dropout, _ = self.dropout_op2(skipx)
        else:
            out = x
            skipx_dropout = skipx
        out = self.relu1(self.bn(self.up_conv(out)))
        xcat = self.cat((out, skipx_dropout))
        out = self.ops(xcat)
        out = self.relu2(mindspore.ops.tensor_add(out, xcat))
        return out


class OutputTransition(nn.Cell):
    """output transition module"""

    def __init__(self, in_channels, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=(5, 5, 5), pad_mode='pad', padding=2)
        self.bn = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=(3, 3, 3), pad_mode='pad', padding=1)
        self.relu = ELUCons(elu, 2)
        self.softmax = nn.Softmax(axis=1)
        self.transpose = mindspore.ops.Transpose()

    def construct(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        B, C, X, Y, Z = out.shape
        out = out.view(B, C, -1)
        out = self.softmax(out)
        out = out[:, 0, :].view(B, X, Y, Z)
        return out


class VNet(nn.Cell):
    """vnet model"""

    def __init__(self, dropout=True, elu=True):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=dropout)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=dropout)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=dropout)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=dropout)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu)

    def construct(self, x):
        """vnet construct"""

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
