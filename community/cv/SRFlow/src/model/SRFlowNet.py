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
The main part of SRFlowNet
"""

import math

import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import stop_gradient


from src.model.RRDBNet import RRDBNet
from src.model.FlowUpsamplerNet import FlowUpsamplerNetFor, FlowUpsamplerNetRev


class SRFlowNetFor(nn.Cell):
    """
    The train part of SRFlowNet
    """
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        super().__init__()
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        self.flowUpsamplerNet = FlowUpsamplerNetFor(image_shape=(3, 160, 160), opt=opt)
        self.scale = opt['scale']
        self.log2 = math.log(2)

    def construct(self, gt=None, lr=None):
        return self.normal_flow(gt, lr)

    def normal_flow(self, gt, lr):
        lr_enc = self.rrdbPreprocessing(lr)
        zeros_like = ops.ZerosLike()
        logdet = zeros_like(gt[:, 0, 0, 0])
        shape = ops.Shape()
        pixels = shape(gt)[2] * shape(gt)[3]
        z, logdet = self.flowUpsamplerNet(gt=gt, logdet=logdet, rrdbResults=lr_enc)
        objective = logdet + self.GaussianDiag_logp(z)
        nll = (-objective) / (self.log2 * pixels)
        return nll

    def GaussianDiag_logp(self, x):
        likelihood = -0.5 * (x ** 2 + np.log(2 * np.pi))
        reduce_sum = ops.ReduceSum()
        log = reduce_sum(likelihood, [1, 2, 3])
        return log

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr)
        return rrdbResults


class SRFlowNetRev(nn.Cell):
    """
    The test part of SRFlowNet
    """
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        super().__init__()
        self.opt = opt
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        self.flowUpsamplerNet = FlowUpsamplerNetRev(image_shape=(3, 160, 160), opt=opt)
        self.scale = opt['scale']

    def construct(self, lr=None):
        x, logdet = self.reverse_flow(lr)
        x = stop_gradient(x)
        logdet = stop_gradient(logdet)
        return x

    def reverse_flow(self, lr):
        zeros_like = ops.ZerosLike()
        logdet = zeros_like(lr[:, 0, 0, 0])
        lr_enc = self.rrdbPreprocessing(lr)
        x, logdet = self.flowUpsamplerNet(logdet=logdet, rrdbResults=lr_enc)
        return x, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr)
        return rrdbResults
