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
"""edsr_slim.py"""
from src import common

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class EDSR(nn.Cell):
    """[EDSR]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, args):
        super(EDSR, self).__init__()

        self.n_colors = args.n_colors
        n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU()
        self.rgb_range = args.rgb_range

        self.sub_mean = common.MeanShift(self.rgb_range)
        self.add_mean = common.MeanShift(self.rgb_range, sign=1)
        self.head = common.conv(args.n_colors, self.n_feats, self.kernel_size, padding=self.kernel_size//2)

        m_body = [
            common.ResidualBlock(
                self.n_feats, self.kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.CellList(m_body)
        self.body_conv = common.conv(self.n_feats, self.n_feats, self.kernel_size, padding=self.kernel_size//2)

        self.upsampler = common.Upsampler(scale, self.n_feats)
        self.tail_conv = common.conv(self.n_feats, args.n_colors, self.kernel_size, padding=self.kernel_size//2)

    def construct(self, x, width_mult=Tensor(1.0)):
        """construct"""
        width_mult = width_mult.asnumpy().item()
        feature_width = int(self.n_feats * width_mult)
        conv2d = ops.Conv2D(out_channel=feature_width, kernel_size=self.kernel_size, mode=1, pad_mode='pad',
                            pad=self.kernel_size // 2)
        biasadd = ops.BiasAdd()

        x = self.sub_mean(x)
        weight = self.head.weight[:feature_width, :self.n_colors, :, :]
        bias = self.head.bias[:feature_width]
        x = conv2d(x, weight)
        x = biasadd(x, bias)

        residual = x
        for block in self.body:
            residual = block(residual, width_mult)
        weight = self.body_conv.weight[:feature_width, :feature_width, :, :]
        bias = self.body_conv.bias[:feature_width]
        residual = conv2d(residual, weight)
        residual = biasadd(residual, bias)
        residual += x

        x = self.upsampler(residual, width_mult)
        weight = self.tail_conv.weight[:self.n_colors, :feature_width, :, :]
        bias = self.tail_conv.bias[:self.n_colors]
        conv2d = ops.Conv2D(out_channel=self.n_colors, kernel_size=self.kernel_size,
                            mode=1, pad_mode='pad', pad=self.kernel_size//2)
        x = conv2d(x, weight)
        x = biasadd(x, bias)
        x = self.add_mean(x)

        return x
