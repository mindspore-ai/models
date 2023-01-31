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

# This file refers to https://github.com/sanghyun-son/EDSR-PyTorch

import numpy as np
import mindspore
import mindspore.common.initializer as init
from mindspore import nn, Parameter

import common_gumbel_softmax_ms as mycommon

convghost = 'ghost'
default_conv = mycommon.default_conv
ghost_conv = mycommon.default_ghost


def _weights_init(m):
    #     print(classname)
    if isinstance(m, (nn.Dense, nn.Conv2d)):
        #         init.kaiming_normal(m.weight)
        init.Zero(m.weight)
        init.Zero(m.bias)


class RgbNormal(nn.Conv2d):
    """
    "MeanShift" in EDSR paper pytorch-code:
    https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/common.py

    it is not unreasonable in the case below
    if std != 1 and sign = -1: y = x * rgb_std - rgb_range * rgb_mean
    if std != 1 and sign =  1: y = x * rgb_std + rgb_range * rgb_mean
    they are not inverse operation for each other!

    so use "RgbNormal" instead, it runs as below:
    if inverse = False: y = (x / rgb_range - mean) / std
    if inverse = True : x = (y * std + mean) * rgb_range
    """

    def __init__(self, rgb_range, rgb_mean, rgb_std, inverse=False):
        super().__init__(3, 3, kernel_size=1, pad_mode='valid', has_bias=True)
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.inverse = inverse
        std = np.array(self.rgb_std, dtype=np.float32)
        mean = np.array(self.rgb_mean, dtype=np.float32)
        if not inverse:
            # y: (x / rgb_range - mean) / std <=> x * (1.0 / rgb_range / std) + (-mean) / std
            weight = (1.0 / self.rgb_range / std).reshape((1, -1, 1, 1))
            bias = (-mean / std)
        else:
            # x: (y * std + mean) * rgb_range <=> y * (std * rgb_range) + mean * rgb_range
            weight = (self.rgb_range * std).reshape((1, -1, 1, 1))
            bias = (mean * rgb_range)

        weight = np.tile(weight, (3, 1, 1, 1))
        # bias = np.tile(bias, (3, 1, 1, 1))

        self.weight = Parameter(weight, requires_grad=False).astype('float16')
        self.bias = Parameter(bias, requires_grad=False).astype('float16')


class EDSR4GhostSRMs(nn.Cell):
    """
        EDSR for GhostSR version
    """
    def __init__(self, scale=2):
        super().__init__()
        n_resblocks = 32
        n_feats = 256
        kernel_size = 3
        act = nn.ReLU()

        self.sub_mean = mycommon.MeanShift(255)
        self.add_mean = mycommon.MeanShift(255, sign=1)

        # self.sub_mean = RgbNormal(rgb_range, rgb_mean, rgb_std, inverse=False)
        # self.add_mean = RgbNormal(rgb_range, rgb_mean, rgb_std, inverse=True)

        # define head module
        m_head = [default_conv(3, n_feats, kernel_size)]

        # define body module
        if convghost == 'ghost':
            m_body = [
                mycommon.GhostResBlock(
                    ghost_conv, n_feats, kernel_size, dir_num=1, act=act, res_scale=0.1
                ) for _ in range(n_resblocks)
            ]
            m_body.append(default_conv(n_feats, n_feats, kernel_size))

        elif convghost == 'conv':
            m_body = [
                mycommon.ConvResBlock(
                    default_conv, n_feats, kernel_size, act=act, res_scale=0.1
                ) for _ in range(n_resblocks)
            ]
            m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            mycommon.Upsampler(default_conv, scale, n_feats, act=False),
            default_conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.SequentialCell(m_head)
        self.body = nn.SequentialCell(m_body)
        self.tail = nn.SequentialCell(m_tail)

    def construct(self, x):
        """
            construct
        """
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_pre_trained_param_dict(self, new_param_dict, strict=True):
        """
        load pre_trained param dict from edsr_x2
        """
        own_param = self.parameters_dict()
        for name, new_param in new_param_dict.items():
            if len(name) >= 4 and name[:4] == "net.":
                name = name[4:]
            if name in own_param:
                if isinstance(new_param, Parameter):
                    param = own_param[name]
                    if tuple(param.data.shape) == tuple(new_param.data.shape):
                        param.set_data(type(param.data)(new_param.data))
                    elif name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_param[name].shape, new_param.shape))
                elif strict:
                    if name.find('tail') == -1:
                        raise KeyError('unexpected key "{}" in parameters_dict()'
                                       .format(name))


if __name__ == '__main__':
    model = EDSR4GhostSRMs().cuda()
    model.load_state_dict(mindspore.load_checkpoint('./model_best.pt'), strict=True)
    print('success')
