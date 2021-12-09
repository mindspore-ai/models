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
The controller of Networks
"""

from src.networks.netwroks_init import init_net
from src.networks.layer_func import get_norm_layer
from src.networks.networks_G import ResnetGenerator, UnetGenerator, PartUnet2Generator
from src.networks.networks_G import PartUnetGenerator, Combiner
from src.networks.networks_D import NLayerDiscriminator, PixelDiscriminator


def define_G(input_nc, output_nc, ngf, netG, norm='batch',
             use_dropout=False, init_type='normal',
             init_gain=0.02, nnG=9):
    """
    define generators
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_nblocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=nnG)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # default for pix2pix
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_ndown':
        net = UnetGenerator(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'partunet':
        net = PartUnetGenerator(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'partunet2':
        net = PartUnet2Generator(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'combiner':
        net = Combiner(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=2)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain)


def define_D(input_nc, ndf, netD, n_layers_D=3,
             norm='batch', use_sigmoid=False,
             init_type='normal', init_gain=0.02):
    """
    define discriminators
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain)
