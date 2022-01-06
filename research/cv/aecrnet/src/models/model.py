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
"""model.py"""
import mindspore
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.ops import operations as P

from src.models.DCN import DeformConv2d
ops_print = ops.Print()
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, pad_mode='pad', padding=(kernel_size // 2), has_bias=bias)

class MeanShift(nn.Conv2d):
    """MeanShift"""
    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):
        """MeanShift"""
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.reshape = P.Reshape()
        self.eye = P.Eye()
        std = Tensor(rgb_std, mstype.float32)
        self.weight.set_data(
            self.reshape(self.eye(3, 3, mstype.float32), (3, 3, 1, 1)) / self.reshape(std, (3, 1, 1, 1)))
        self.weight.requires_grad = False
        self.bias = Parameter(
            sign * rgb_range * Tensor(rgb_mean, mstype.float32) / std, name='bias', requires_grad=False)
        self.has_bias = True


class AdaptiveAvgPool2d(nn.Cell):
    """rcan"""
    def __init__(self):
        """rcan"""
        super().__init__()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        """rcan"""
        return self.reduce_mean(x, 0)

class PALayer(nn.Cell):
    """[PALayer]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.SequentialCell([
            nn.Conv2d(channel, channel // 8, 1, padding=0, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, 1, padding=0, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        """construct"""
        y = self.pa(x)
        return x * y

class CALayer(nn.Cell):
    """[CALayer]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.ca = nn.SequentialCell([
            nn.Conv2d(channel, channel // 8, 1, padding=0, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        """construct"""
        avg_pool = nn.AvgPool2d((x.shape[-2], x.shape[-1]))
        y = avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Cell):
    """[DehazeBlock]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def construct(self, x):
        """construct"""
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Mix(nn.Cell):
    """[Mix]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = Parameter(Tensor(m, mindspore.float32), requires_grad=True)
        self.w = w
        self.exp = ops.Exp()
        self.mix_block = nn.Sigmoid()

    def construct(self, fea1, fea2):
        """construct"""
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out, self.w


class Dehaze(nn.Cell):
    """[Dehaze]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_nc, output_nc, ngf=64, rgb_range=255):
        super(Dehaze, self).__init__()

        rgb_mean = (0.58726025, 0.59813744, 0.63799095)
        rgb_std = (0.18993913, 0.18339698, 0.17736082)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.down1 = nn.SequentialCell([
            nn.Conv2d(input_nc, ngf, kernel_size=7, pad_mode='pad', padding=3, has_bias=True),
        ])
        self.down2 = nn.SequentialCell([
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True),
        ])
        self.down3 = nn.SequentialCell([
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True),
        ])
        self.block = DehazeBlock(default_conv, ngf*4, 3)
        self.up1 = nn.SequentialCell([
            nn.Conv2dTranspose(ngf*4, ngf*2, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True),
            nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1))),
        ])
        self.up2 = nn.SequentialCell([
            nn.Conv2dTranspose(ngf*2, ngf, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True),
            nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1))),
        ])
        self.up3 = nn.SequentialCell([
            nn.Conv2d(ngf, output_nc, kernel_size=7, pad_mode='pad', padding=3, has_bias=True),
        ])
        self.dcn_block = DeformConv2d(256, 256)
        self.deconv = nn.Conv2d(3, 3, 3, stride=1, padding=1, pad_mode='pad', has_bias=True)

        m1 = -1
        m2 = -0.6
        self.mix4 = Mix(m=m1)
        self.mix5 = Mix(m=m2)
        print(f'Mix setting: m1={m1} m2={m2}')

    def construct(self, x):
        """construct"""
        x = self.sub_mean(x)
        x_deconv = self.deconv(x)
        x_down1 = self.down1(x_deconv)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x1 = self.block(x_down3)
        x2 = self.block(x1)

        x3 = self.block(x2)
        x4 = self.block(x3)

        x5 = self.block(x4)
        x6 = self.block(x5)

        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix, _ = self.mix4(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix)
        x_up1_mix, _ = self.mix5(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix)
        out = self.up3(x_up2)

        out = self.add_mean(out)
        return out
