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
"""
The Upsampler Layer of Glow
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from src.model.FlowStep import FlowStepFor, FlowStepRev, FlowStepCondFor, FlowStepCondRev
from src.model.Split import Split2dFor, Split2dRev
from src.model.Flow import SqueezeLayerFor, SqueezeLayerRev


class FlowUpsamplerNetFor(nn.Cell):
    """
    The train part of Upsampler Layer
    """
    def __init__(self, image_shape, opt=None):

        super().__init__()
        self.opt = opt
        self.LevelList = []
        self.L = opt['network_G']['flow']['L']
        self.K = opt['network_G']['flow']['K']
        self.C, H, W = image_shape

        self.layers_1 = nn.CellList()
        self.layers_2 = nn.CellList()
        self.layers_3 = nn.CellList()

        H, W = self.arch_squeeze(H, W, self.layers_1)
        self.arch_additionalFlowAffine(self.layers_1)
        self.arch_FlowStep(self.K, self.layers_1)
        self.arch_split(1, self.L, opt, self.layers_1)

        H, W = self.arch_squeeze(H, W, self.layers_2)
        self.arch_additionalFlowAffine(self.layers_2)
        self.arch_FlowStep(self.K, self.layers_2)
        self.arch_split(2, self.L, opt, self.layers_2)

        H, W = self.arch_squeeze(H, W, self.layers_3)
        self.arch_additionalFlowAffine(self.layers_3)
        self.arch_FlowStep(self.K, self.layers_3)
        self.arch_split(3, self.L, opt, self.layers_3)

        self.H = H
        self.W = W
        self.scaleH = 160 / H
        self.scaleW = 160 / W

    def add_layer(self, layers, layer):
        layers.append(layer)

    def arch_squeeze(self, H, W, layers):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.add_layer(layers, SqueezeLayerFor(factor=2))
        return H, W

    def arch_additionalFlowAffine(self, layers):
        n_additionalFlowNoAffine = int(self.opt['network_G']['flow']['additionalFlowNoAffine'])
        for _ in range(n_additionalFlowNoAffine):
            self.add_layer(layers, FlowStepFor(in_channels=self.C))

    def arch_FlowStep(self, K, layers):
        for _ in range(K):
            self.add_layer(layers, FlowStepCondFor(in_channels=self.C))

    def arch_split(self, L, levels, opt, layers):
        if opt['network_G']['flow']['split']['enable'] is True and L < levels - 1:
            consume_ratio = opt['network_G']['flow']['split']['consume_ratio']
            split = Split2dFor(num_channels=self.C, consume_ratio=consume_ratio, opt=opt)
            self.add_layer(layers, split)
            self.C = self.C - int(round(self.C * consume_ratio))

    def construct(self, gt=None, logdet=0., rrdbResults=None):
        z, logdet = self.encode(gt, rrdbResults, logdet=logdet)
        return z, logdet

    def encode(self, gt, rrdbResults, logdet=0.0):
        z = gt
        for layer in self.layers_1:
            z, logdet = layer(z, logdet, rrdbResults=rrdbResults['fea_up2'])
        for layer in self.layers_2:
            z, logdet = layer(z, logdet, rrdbResults=rrdbResults['fea_up1'])
        for layer in self.layers_3:
            z, logdet = layer(z, logdet, rrdbResults=rrdbResults['fea_up0'])

        return z, logdet


class FlowUpsamplerNetRev(nn.Cell):
    """
    The train part of Upsampler Layer
    """
    def __init__(self, image_shape, opt=None):

        super().__init__()
        self.opt = opt
        self.L = opt['network_G']['flow']['L']
        self.K = opt['network_G']['flow']['K']
        self.C, H, W = image_shape

        self.layers_1 = nn.CellList()
        self.layers_2 = nn.CellList()
        self.layers_3 = nn.CellList()

        H, W = self.arch_squeeze(H, W, self.layers_1)
        self.arch_additionalFlowAffine(self.layers_1)
        self.arch_FlowStep(self.K, self.layers_1)
        self.arch_split(1, self.L, opt, self.layers_1)

        H, W = self.arch_squeeze(H, W, self.layers_2)
        self.arch_additionalFlowAffine(self.layers_2)
        self.arch_FlowStep(self.K, self.layers_2)
        self.arch_split(2, self.L, opt, self.layers_2)

        H, W = self.arch_squeeze(H, W, self.layers_3)
        self.arch_additionalFlowAffine(self.layers_3)
        self.arch_FlowStep(self.K, self.layers_3)
        self.arch_split(3, self.L, opt, self.layers_3)

        self.H = H
        self.W = W
        self.scaleH = 160 / H
        self.scaleW = 160 / W

        self.heat = Tensor(self.opt['heat'], mindspore.float32)
        self.mean = Tensor(0, mindspore.float32)

        self.z = ops.normal(mean=self.mean, stddev=self.heat, shape=(1, 96, 20, 20))

    def add_layer(self, layers, layer):
        if layers:
            layers.insert(0, layer)
        else:
            layers.append(layer)

    def arch_squeeze(self, H, W, layers):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.add_layer(layers, SqueezeLayerRev(factor=2))
        return H, W

    def arch_additionalFlowAffine(self, layers):
        n_additionalFlowNoAffine = int(self.opt['network_G']['flow']['additionalFlowNoAffine'])
        for _ in range(n_additionalFlowNoAffine):
            self.add_layer(layers, FlowStepRev(in_channels=self.C))

    def arch_FlowStep(self, K, layers):
        for _ in range(K):
            self.add_layer(layers, FlowStepCondRev(in_channels=self.C))

    def arch_split(self, L, levels, opt, layers):
        if opt['network_G']['flow']['split']['enable'] is True and L < levels - 1:
            consume_ratio = opt['network_G']['flow']['split']['consume_ratio']
            split = Split2dRev(num_channels=self.C, consume_ratio=consume_ratio, opt=opt)
            self.add_layer(layers, split)
            self.C = self.C - int(round(self.C * consume_ratio))

    def construct(self, logdet=0., rrdbResults=None):
        z, logdet = self.decode(rrdbResults, logdet=logdet)
        return z, logdet

    def decode(self, rrdbResults, logdet=0.0):
        """
        decode
        """
        sr = self.z

        for layer in self.layers_3:
            sr, logdet = layer(sr, logdet, rrdbResults=rrdbResults['fea_up0'])
        for layer in self.layers_2:
            sr, logdet = layer(sr, logdet, rrdbResults=rrdbResults['fea_up1'])
        for layer in self.layers_1:
            sr, logdet = layer(sr, logdet, rrdbResults=rrdbResults['fea_up2'])

        return sr, logdet
