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
""" SPADE Discriminator """


import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
import mindspore as ms
import mindspore.ops as ops
from src.util.instancenorm import InstanceNorm2d
from src.models.spectral_norm import SpectualNormConv2d
from src.models.init_Parameter import XavierNormal


class MultiscaleDiscriminator(nn.Cell):
    def __init__(self, opt):
        super(MultiscaleDiscriminator, self).__init__()
        self.opt = opt
        self.avgpool_op = ops.AvgPool(pad_mode="same", kernel_size=3, strides=2)
        subnetD1 = self.create_single_discriminator(opt)
        subnetD2 = self.create_single_discriminator(opt)
        self.discriminator_0 = subnetD1
        self.discriminator_1 = subnetD2

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def construct(self, input_obj):
        result_0 = self.discriminator_0(input_obj)
        input_obj = self.avgpool_op(input_obj)
        result_1 = self.discriminator_1(input_obj)
        result = (result_0, result_1)
        return result

class NLayerDiscriminator(nn.Cell):
    def __init__(self, opt):
        super(NLayerDiscriminator, self).__init__()
        self.opt = opt
        self.assign = ops.Assign()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        xaviernormal = XavierNormal(0.02)
        weight_0 = xaviernormal.initialize([nf, input_nc, kw, kw])
        sequence = [[
            nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, has_bias=True, padding=padw, pad_mode='pad',
                      weight_init=Tensor(weight_0, ms.float32)),
            nn.LeakyReLU()]]
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            weight = Tensor(xaviernormal.initialize([nf, nf_prev, kw, kw]), ms.float32)
            conv = SpectualNormConv2d(nf_prev, nf, kernel_size=kw, has_bias=False, \
                                      weight_init=weight, padding=padw, stride=stride, pad_mode='pad')
            sequence += [[conv,
                          InstanceNorm2d(nf, affine=False),
                          nn.LeakyReLU()
                          ]]
        weight = xaviernormal.initialize([1, nf, kw, kw])
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, pad_mode="pad", has_bias=True,
                                weight_init=Tensor(weight, ms.float32))]]
        self.model0 = nn.SequentialCell(*sequence[0])
        self.model1_conv = sequence[1][0]
        self.model1_bn = sequence[1][1]
        self.model1_relu = sequence[1][2]
        self.model2_conv = sequence[2][0]
        self.model2_bn = sequence[2][1]
        self.model2_relu = sequence[2][2]
        self.model3_conv = sequence[3][0]
        self.model3_bn = sequence[3][1]
        self.model3_relu = sequence[3][2]
        self.model4 = nn.SequentialCell(sequence[4])

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def construct(self, input_tensor):
        intermediate_output_0 = self.model0(input_tensor)
        intermediate_output_1 = self.model1_relu(self.model1_bn(self.model1_conv(intermediate_output_0)))
        intermediate_output_2 = self.model2_relu(self.model2_bn(self.model2_conv(intermediate_output_1)))
        intermediate_output_3 = self.model3_relu(self.model3_bn(self.model3_conv(intermediate_output_2)))
        intermediate_output_4 = self.model4(intermediate_output_3)
        results = (intermediate_output_0, intermediate_output_1, intermediate_output_2, intermediate_output_3,
                   intermediate_output_4)
        return results
