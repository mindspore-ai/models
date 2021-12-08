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
# ===========================================================================
"""Atrous spatial pyramid pooling module"""
import mindspore.nn as nn
import mindspore.ops as ops

from ..modules.bn import NormLeakyReLU


class ASPP(nn.Cell):
    """Atrous spatial pyramid pooling"""
    def __init__(self,
                 in_channels,
                 multi=1,
                 momentum=0.9,
                 eps=1e-5,
                 parallel=True):
        super(ASPP, self).__init__()
        out_channels = 256
        self.global_pooling = ops.ReduceMean(keep_dims=True)
        self.upsample = nn.ResizeBilinear()
        self.cat = ops.Concat(axis=1)

        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                               has_bias=False, weight_init='HeNormal')
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(6 * multi), pad_mode='pad', padding=int(6 * multi),
                               has_bias=False, weight_init='HeNormal')
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(12 * multi), pad_mode='pad', padding=int(12 * multi),
                               has_bias=False, weight_init='HeNormal')
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(18 * multi), pad_mode='pad', padding=int(18 * multi),
                               has_bias=False, weight_init='HeNormal')
        self.aspp5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                               has_bias=False, weight_init='HeNormal')

        self.aspp1_bn = NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)
        self.aspp2_bn = NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)
        self.aspp3_bn = NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)
        self.aspp4_bn = NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)
        self.aspp5_bn = NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)

        self.conv2 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1,
                               has_bias=False, weight_init='HeNormal')
        self.bn2 = NormLeakyReLU(out_channels, momentum, eps, parallel=parallel)

    def construct(self, x):
        """construct"""
        x0_ = self.aspp1(x)
        x0 = self.aspp1_bn(x0_)

        x1_ = self.aspp2(x)
        x1 = self.aspp2_bn(x1_)

        x2_ = self.aspp3(x)
        x2 = self.aspp3_bn(x2_)

        x3_ = self.aspp4(x)
        x3 = self.aspp4_bn(x3_)

        x4_0 = self.global_pooling(x, (-2, -1))
        x4_1 = self.aspp5(x4_0)
        x4_2 = self.aspp5_bn(x4_1)
        x4 = self.upsample(x4_2, (x.shape[2], x.shape[3]), None, True)

        x5 = self.cat((x0, x1, x2, x3, x4))
        x6 = self.conv2(x5)
        output = self.bn2(x6)

        return output
