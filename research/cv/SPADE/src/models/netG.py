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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" SPADE Generator """

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from src.models.init_Parameter import XavierNormal
from src.models.architecture import SPADEResnetBlock

def compute_latent_vector_size(opt):
    if opt.num_upsampling_layers == 'normal':
        num_up_layers = 5
    elif opt.num_upsampling_layers == 'more':
        num_up_layers = 6
    elif opt.num_upsampling_layers == 'most':
        num_up_layers = 7
    else:
        raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                         opt.num_upsampling_layers)

    sw = opt.crop_size // (2 ** num_up_layers)
    sh = round(sw / opt.aspect_ratio)

    return sw, sh


class SPADEGenerator(nn.Cell):
    def __init__(self, opt):
        super(SPADEGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = compute_latent_vector_size(opt)
        self.interpolate = ops.ResizeNearestNeighbor((self.sh, self.sw))
        xaviernormal = XavierNormal(0.02)
        weight = xaviernormal.initialize([16 * nf, self.opt.semantic_nc, 3, 3])
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1, pad_mode='pad', \
                            has_bias=True, weight_init=Tensor(weight, ms.float32), bias_init="zeros")
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        final_nc = nf
        weight_conv_img = xaviernormal.initialize([3, final_nc, 3, 3])
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1, pad_mode='pad', has_bias=True,
                                  weight_init=Tensor(weight_conv_img, ms.float32), bias_init="zeros")
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def construct(self, input_obj):
        seg = input_obj
        x = self.interpolate(seg)
        x = self.fc(x)
        x = self.head_0(x, seg)
        x = ops.ResizeNearestNeighbor((self.sh*2, self.sw*2))(x)

        x = self.G_middle_0(x, seg)

        x = self.G_middle_1(x, seg)

        x = ops.ResizeNearestNeighbor((self.sh*4, self.sw*4))(x)
        x = self.up_0(x, seg)
        x = ops.ResizeNearestNeighbor((self.sh*8, self.sw*8))(x)
        x = self.up_1(x, seg)
        x = ops.ResizeNearestNeighbor((self.sh*16, self.sw*16))(x)
        x = self.up_2(x, seg)
        x = ops.ResizeNearestNeighbor((self.sh*32, self.sw*32))(x)
        x = self.up_3(x, seg)
        x = self.conv_img(self.leaky_relu(x))
        x = self.tanh(x)
        return x
