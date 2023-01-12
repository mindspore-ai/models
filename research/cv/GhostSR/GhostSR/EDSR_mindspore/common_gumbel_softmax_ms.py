# 2022.12.27-Changed for EDSR-PyTorch
#      Huawei Technologies Co., Ltd. <wangchengcheng11@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
# Copyright 2018 sanghyun-son (https://github.com/sanghyun-son/EDSR-PyTorch).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import numpy as np

import mindspore as ms
from mindspore import nn
import mindspore.common.initializer as init
from GhostSR.unsupported_model.PixelShuffle import PixelShuffle



def exponential_decay(t, _init=10, m=200, finish=1e-2):
    alpha = np.log(_init / finish) / m
    l = - np.log(_init) / alpha
    decay = np.exp(-alpha * (t + l))
    return decay


def sample_gumbel(size, eps=1e-20):
    # print('size.dtype: ', size[0].dtype)
    uniform_real = ms.ops.UniformReal()
    U = uniform_real(size)
    return -ms.ops.log(-ms.ops.log(U + eps) + eps)


def gumbel_softmax(weights, epoch):
    noise_temp = 0.97 ** (epoch - 1)
    noise = sample_gumbel(weights.shape) * noise_temp
    y = weights + noise
    y_abs = y.abs().view(1, -1)
    y_hard = ms.ops.zeros_like(y_abs)
    y_hard[0, ms.ops.Argmax()(y_abs)] = 1
    y_hard = y_hard.view(weights.shape)
    # ret = (y_hard - weights).detach() + weights
    ret = ms.ops.stop_gradient(y_hard - weights) + weights
    return ret


def hard_softmax(weights):
    y_abs = weights.abs().view(1, -1)
    y_hard = ms.ops.ZerosLike()(y_abs)
    y_hard[0, ms.ops.Argmax()(y_abs)] = 1
    y_hard = y_hard.view(weights.shape)
    return y_hard


# 1*1*3*3 shift
class ShiftConvGeneral(nn.Cell):
    def __init__(self, act_channel, in_channels=1, out_channels=1, kernel_size=3, stride=1,
                 padding=1, groups=1,
                 bias=False):
        super(ShiftConvGeneral, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.epoch = 1
        self.act_channel = act_channel
        # self.w_out_channels = in_channels // groups
        self.kernel_size = kernel_size
        self.weight = ms.Parameter(
            ms.Tensor(shape=(out_channels, in_channels // groups, kernel_size, kernel_size),
                      dtype=ms.float16,
                      init=init.HeUniform(negative_slope=math.sqrt(5))), requires_grad=True)
        if bias:
            self.b = ms.Parameter(ms.ops.Zeros(act_channel), requires_grad=True)
        # self.reset_parameters()

    def reset_parameters(self):
        init.HeUniform(self.weight, a=math.sqrt(5))

    def construct(self, x):
        assert x.shape[1] == self.act_channel
        if self.training:
            w = gumbel_softmax(self.weight, self.epoch)
        else:
            w = hard_softmax(self.weight)
        w = w.astype(x.dtype)
        w = ms.numpy.tile(w, (x.shape[1], 1, 1, 1))
        out = ms.ops.Conv2D(self.act_channel, self.kernel_size, stride=self.stride,
                            pad=self.padding, dilation=1,
                            pad_mode='pad', group=x.shape[1])(x, w)
        if self.bias:
            out += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return out


# 统一方向shift, lamda!=0.5
class GhostModule(nn.Cell):
    def __init__(self, inp, oup, kernel_size, dir_num, ratio=0.5, stride=1, bias=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup * ratio)
        new_channels = oup - init_channels

        self.primary_conv = nn.Conv2d(inp, init_channels, kernel_size, stride,
                                      pad_mode='pad', padding=kernel_size // 2, has_bias=bias)
        self.cheap_conv = ShiftConvGeneral(new_channels, 1, 1, kernel_size=3, stride=1, padding=1,
                                           groups=1, bias=False)
        self.concat = ms.ops.Concat(axis=1)

        self.init_channels = init_channels
        self.new_channels = new_channels

    def construct(self, x):
        if self.init_channels > self.new_channels:
            x1 = self.primary_conv(x)
            x2 = self.cheap_conv(x1[:, :self.new_channels, :, :])
        elif self.init_channels == self.new_channels:
            x1 = self.primary_conv(x)
            x2 = self.cheap_conv(x1)
        # elif self.init_channels < self.new_channels:
        else:
            x1 = self.primary_conv(x)
            x1 = x1.repeat(1, 3, 1, 1)
            x2 = self.cheap_conv(x1[:, :self.new_channels, :, :])
        out = self.concat([x1, x2])
        return out[:, :self.oup, :, :]


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        pad_mode='pad', padding=(kernel_size // 2), has_bias=bias)


def default_ghost(in_channels, out_channels, kernel_size, dir_num, bias=True):
    return GhostModule(
        in_channels, out_channels, kernel_size, dir_num, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1, pad_mode='valid', has_bias=True)
        std = ms.Tensor(rgb_std)
        self.weight.set_data(ms.ops.eye(3, 3, ms.float32).view(3, 3, 1, 1) / std.view(3, 1, 1, 1))

        self.bias.set_data(sign * rgb_range * ms.Tensor(rgb_mean) / std)
        for p in self.get_parameters():
            p.requires_grad = False


class GhostResBlock(nn.Cell):
    def __init__(self, conv, n_feats, kernel_size, dir_num=1, bias=True, bn=False, act=nn.ReLU(),
                 res_scale=1):
        super(GhostResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, dir_num=dir_num, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats, momentum=0.9))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(m)
        self.mul = ms.ops.Mul()
        self.res_scale = res_scale

    def construct(self, x):
        res = self.mul(self.body(x), self.res_scale)
        # res = self.body(x) * self.res_scale
        res += x
        return res


class ConvResBlock(nn.Cell):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(), res_scale=1):
        super(ConvResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats, momentum=0.9))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(m)
        self.res_scale = ms.Tensor(res_scale, dtype=ms.int32)

    def construct(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.SequentialCell):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats, momentum=0.9))
                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats, momentum=0.9))
            if act == 'relu':
                m.append(nn.ReLU())
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(m)
