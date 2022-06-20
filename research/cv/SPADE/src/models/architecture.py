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
""" SPADE Resnet Block """

import mindspore.nn as nn
from mindspore import Tensor, ops
import mindspore as ms
from src.models.normalization import SPADE
from src.models.init_Parameter import XavierNormal
from src.models.spectral_norm import SpectualNormConv2d

class SPADEResnetBlock(nn.Cell):
    def __init__(self, fin, fout, opt):
        super(SPADEResnetBlock, self).__init__()
        self.assign = ops.Assign()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        xaviernormal = XavierNormal(0.02)
        conv_0_weight = Tensor(xaviernormal.initialize([fmiddle, fin, 3, 3]), ms.float32)
        conv_1_weight = Tensor(xaviernormal.initialize([fout, fmiddle, 3, 3]), ms.float32)

        self.conv_0 = SpectualNormConv2d(fin, fmiddle, 3, has_bias=True, padding=1, \
                                         pad_mode='pad', weight_init=conv_0_weight)
        self.conv_1 = SpectualNormConv2d(fmiddle, fout, 3, has_bias=True, padding=1, \
                                         pad_mode='pad', weight_init=conv_1_weight)
        if self.learned_shortcut:
            conv_s_weight = Tensor(xaviernormal.initialize([fout, fin, 1, 1]), ms.float32)
            self.conv_s = SpectualNormConv2d(fin, fout, kernel_size=1, has_bias=False, weight_init=conv_s_weight)
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, opt.distribute)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, opt.distribute)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, opt.distribute)
        self.leaky_relu = nn.LeakyReLU()

    def construct(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.leaky_relu(self.norm_0(x, seg)))
        dx = self.conv_1(self.leaky_relu(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s
