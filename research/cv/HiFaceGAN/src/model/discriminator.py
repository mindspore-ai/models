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
"""Multiscale discriminator for HiFaceGAN network"""
import mindspore.nn as nn

from src.model.architecture import Conv2dNormalized


class MultiscaleDiscriminator(nn.Cell):
    """Multiscale discriminator"""
    def __init__(self, ndf, input_nc, use_gan_feat_loss):
        super().__init__()
        self.num_D = 2
        self.D1 = NLayerDiscriminator(ndf, input_nc, use_gan_feat_loss)
        self.D2 = NLayerDiscriminator(ndf, input_nc, use_gan_feat_loss)
        self.avg_pool = nn.AvgPool2d(3, 2, pad_mode='same')

    def construct(self, x):
        """Feed forward"""
        out1 = self.D1(x)
        out2 = self.D2(self.avg_pool(x))
        return out1, out2


class NLayerDiscriminator(nn.Cell):
    """Inner discriminator with N layers"""
    def __init__(self, ndf, input_nc, use_gan_feat_loss):
        super().__init__()
        self.use_gan_feat_loss = use_gan_feat_loss
        kw = 4
        padw = 2
        input_nc = input_nc + 3
        n_layers_D = 4

        self.model = nn.CellList([
            nn.SequentialCell([
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, pad_mode='pad'),
                nn.LeakyReLU(0.2)
            ])
        ])

        for n in range(1, n_layers_D):
            nf_prev = ndf
            ndf = min(ndf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence = nn.SequentialCell([
                Conv2dNormalized(nf_prev, ndf, kernel_size=kw, stride=stride,
                                 pad_mode='pad', pad=padw),
                nn.InstanceNorm2d(ndf),
                nn.LeakyReLU(0.2)
            ])
            self.model.append(sequence)

        self.model.append(
            nn.Conv2d(ndf, 1, kernel_size=kw, stride=1, padding=padw, pad_mode='pad')
        )

    def construct(self, x):
        """Feed forward"""
        out_0 = self.model[0](x)
        out_1 = self.model[1](out_0)
        out_2 = self.model[2](out_1)
        out_3 = self.model[3](out_2)
        out_4 = self.model[4](out_3)
        if self.use_gan_feat_loss:
            return [out_0, out_1, out_2, out_3, out_4]
        return [out_4]
