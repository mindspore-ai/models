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
Discriminator Network
"""

import functools
import mindspore.nn as nn


class NLayerDiscriminator(nn.Cell):
    """NLayerDiscriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kw, 2, padding=padw, pad_mode="pad"),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, has_bias=use_bias, pad_mode="pad"),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, has_bias=use_bias, pad_mode="pad"),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, pad_mode="pad")]

        if use_sigmoid:  # no_lsgan, use sigmoid before calculating bceloss(binary cross entropy)
            sequence.append(nn.Sigmoid())

        self.model = nn.SequentialCell(sequence)

    def construct(self, input_data):
        predict = self.model(input_data)
        predict = predict.view(predict.shape[0], -1)
        return predict


class PixelDiscriminator(nn.Cell):
    """PixelDiscriminator"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, pad_mode="pad"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, has_bias=use_bias, pad_mode="pad"),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, has_bias=use_bias, pad_mode="pad")
        ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.SequentialCell(self.net)

    def construct(self, input_data):
        predict = self.net(input_data)
        predict = predict.view(predict.shape[0], -1)
        return predict
