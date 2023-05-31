# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore.common import initializer as init


class Conv(nn.Cell):
    """2D convolution w/ MSRA init."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            pad_mode="pad",
            stride=stride,
            padding=padding,
            dilation=dilation,
            has_bias=bias,
        )

        init_gain = 0.1
        self.conv.weight.set_data(
            init.initializer(init.HeNormal(init_gain), self.conv.weight.shape)
        )
        if bias:
            self.conv.bias.set_data(init.initializer(0.001, self.conv.bias.shape))

    def construct(self, x):
        return self.conv(x)


class DenseConv(nn.Cell):
    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def construct(self, x):
        feat = self.act(self.conv1(x))
        out = ops.concat([x, feat], axis=1)
        return out


class SRUnit(nn.Cell):
    def __init__(self, mode, nf, upscale, out_c=1):
        super(SRUnit, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale

        if mode == "2x2":
            self.conv1 = Conv(1, nf, 2)
        elif mode == "2x2d":
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == "1x4":
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, out_c * upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def construct(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = ops.tanh(self.conv6(x))
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        return x


class SRNet(nn.Cell):
    def __init__(self, mode, nf=64, upscale=None, out_c=1):
        super(SRNet, self).__init__()
        self.mode = mode

        if "x1" in mode:
            assert upscale is None
        if mode == "Sx1":
            self.model = SRUnit("2x2", nf, upscale=1, out_c=1)
            self.k = 2
            self.s = 1
        elif mode == "SxN":
            self.model = SRUnit("2x2", nf, upscale=upscale, out_c=out_c)
            self.k = 2
            self.s = upscale
        elif mode == "Dx1":
            self.model = SRUnit("2x2d", nf, upscale=1, out_c=1)
            self.k = 3
            self.s = 1
        elif mode == "DxN":
            self.model = SRUnit("2x2d", nf, upscale=upscale, out_c=out_c)
            self.k = 3
            self.s = upscale
        elif mode == "Yx1":
            self.model = SRUnit("1x4", nf, upscale=1, out_c=1)
            self.k = 3
            self.s = 1
        elif mode == "YxN":
            self.model = SRUnit("1x4", nf, upscale=upscale, out_c=out_c)
            self.k = 3
            self.s = upscale
        elif mode == "Cx1":
            self.model = SRUnit("1x4", nf, upscale=1, out_c=1)
            self.k = 4
            self.s = 1
        elif mode == "CxN":
            self.model = SRUnit("1x4", nf, upscale=upscale, out_c=out_c)
            self.k = 4
            self.s = upscale
        elif mode == "Tx1":
            self.model = SRUnit("1x4", nf, upscale=1, out_c=1)
            self.k = 4
            self.s = 1
        elif mode == "TxN":
            self.model = SRUnit("1x4", nf, upscale=upscale, out_c=out_c)
            self.k = 4
            self.s = upscale
        else:
            raise AttributeError
        self.p = self.k - 1

        self.unfold = nn.Unfold(
            ksizes=[1, self.k, self.k, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1]
        )
        self.out_c = out_c

    def construct(self, x):
        b, c, h, w = x.shape
        x = self.unfold(x)  # B,C*K*K,L
        x = x.view(b, c, self.k * self.k, (h - self.p) * (w - self.p))  # B,C,K*K,L
        x = ops.transpose(x, (0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(b * c * (h - self.p) * (w - self.p), self.k, self.k)  # B*C*L,K,K
        x = ops.expand_dims(x, 1)  # B*C*L,1,K,K

        if "Y" in self.mode:
            x = ops.concat(
                [x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 1, 2], x[:, :, 2, 1]], axis=1
            )

            x = ops.expand_dims(ops.expand_dims(x, 1), 1)
        elif "C" in self.mode:
            x = ops.concat(
                [x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2], x[:, :, 0, 3]], axis=1
            )

            x = ops.expand_dims(ops.expand_dims(x, 1), 1)
        elif "T" in self.mode:
            x = ops.concat(
                [x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2], x[:, :, 3, 3]], axis=1
            )

            x = ops.expand_dims(ops.expand_dims(x, 1), 1)

        x = self.model(x)  # B*C*L,out_c,K,K
        x = x.reshape(b, c, (h - self.p), (w - self.p), self.out_c)  # B,C,L,K*K*out_c
        x = ops.transpose(x, (0, 1, 4, 2, 3))  # B,C,K*K*out_c,L
        x = x.reshape(b, c * self.out_c, (h - self.p), (w - self.p))  # B,C*out_c, K*K,L
        return x
