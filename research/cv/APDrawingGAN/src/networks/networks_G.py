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
"""
Generator Network
"""

import functools
import mindspore.nn as nn
from src.networks.networks_block import ResnetBlock, UnetSkipConnectionBlock


class ResnetGenerator(nn.Cell):
    """ResnetGenerator"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=padding_type.upper()),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           has_bias=use_bias, pad_mode="pad"),
                 norm_layer(ngf),
                 nn.ReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, has_bias=use_bias, pad_mode="pad"),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU()]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.Conv2dTranspose(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1,
                                         has_bias=use_bias, pad_mode="pad"),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU()]
        model += [nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=padding_type.upper())]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, pad_mode="pad")]
        model += [nn.Tanh()]

        self.model = nn.SequentialCell(model)

    def construct(self, input_data):
        return self.model(input_data)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
class UnetGenerator(nn.Cell):
    """UnetGenerator"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, \
                                             innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, \
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, \
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, \
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, \
                                             norm_layer=norm_layer)

        self.model = unet_block

    def construct(self, input_data):
        return self.model(input_data)


class PartUnetGenerator(nn.Cell):
    """PartUnetGenerator"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PartUnetGenerator, self).__init__()

        # construct unet structure
        # 3 downs
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, \
                                             innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, \
                                             norm_layer=norm_layer)

        self.model = unet_block

    def construct(self, input_data):
        return self.model(input_data)


class PartUnet2Generator(nn.Cell):
    """PartUnet2Generator"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PartUnet2Generator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, \
                                             innermost=True)
        for _ in range(num_downs - 3):
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=unet_block, \
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, \
                                             norm_layer=norm_layer)

        self.model = unet_block

    def construct(self, input_data):
        return self.model(input_data)


class Combiner(nn.Cell):
    """Combiner"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, \
                 padding_type='constant'):
        assert n_blocks >= 0
        super(Combiner, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=padding_type.upper()),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           has_bias=use_bias, pad_mode="pad"),
                 norm_layer(ngf),
                 nn.ReLU()]

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, \
                                  use_bias=use_bias)]

        model += [nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=padding_type.upper())]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, pad_mode="pad")]
        model += [nn.Tanh()]

        self.model = nn.SequentialCell(model)

    def construct(self, input_data):
        return self.model(input_data)
