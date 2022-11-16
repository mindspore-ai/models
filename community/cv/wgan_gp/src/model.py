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
"""create model"""

from src.dcgan_model import DcganD, DcganG
from src.resgan_model import GoodGenerator, GoodDiscriminator, ResnetGenerator, ResnetDiscriminator



def create_G(model_type, isize, nz, nc, ngf):
    """create generator """
    if model_type == 'resnet':
        if isize == 64:
            netG = GoodGenerator(isize, nz, nc, ngf)
        elif isize == 128:
            netG = ResnetGenerator(isize, nz, nc, ngf)
        else:
            raise Exception('invalid resample value')
    elif model_type == 'dcgan':
        netG = DcganG(isize, nz, nc, ngf)
    else:
        raise Exception('invalid resample value')

    return netG

def create_D(model_type, dataset, isize, nc, ngf):
    """create discriminator """
    if model_type == 'resnet':
        if isize == 64:
            netD = GoodDiscriminator(isize, nc, ngf)
        elif isize == 128:
            netD = ResnetDiscriminator(isize, nc, ngf)
        else:
            raise Exception('invalid resample value')
    elif model_type == 'dcgan':
        if dataset == 'lsun':
            netD = DcganD(isize, nc, ngf, normalization_d=True)
        elif dataset == 'cifar10':
            netD = DcganD(isize, nc, ngf, normalization_d=False)
        else:
            raise Exception('invalid resample value')
    else:
        raise Exception('invalid resample value')

    return netD
