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
'''the loss function'''
from mindspore import nn
import mindspore.ops.functional as F

class DisWithLossCell(nn.Cell):
    '''DisWithLossCell'''

    def __init__(self, netG, netD, loss_fn, auto_prefix=True):
        super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = loss_fn

    def construct(self, real_data, latent_code1):
        fake_data = self.netG(latent_code1)
        real_out = self.netD(real_data)
        real_loss = self.loss_fn(real_out, F.ones_like(real_out))
        fake_out = self.netD(fake_data)
        fake_loss = self.loss_fn(fake_out, F.zeros_like(fake_out))
        loss_D = real_loss + fake_loss

        return loss_D


class GenWithLossCell(nn.Cell):
    '''GenWithLossCell'''

    def __init__(self, netG, netD, loss_fn, auto_prefix=True):
        super(GenWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = loss_fn

    def construct(self, latent_code2):
        fake_data = self.netG(latent_code2)
        fake_out = self.netD(fake_data)
        loss_G = self.loss_fn(fake_out, F.ones_like(fake_out))
        return loss_G
