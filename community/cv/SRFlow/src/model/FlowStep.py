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
The Step Layer of Glow
"""

import mindspore.nn as nn

from src.model.InvertibleConv1x1 import InvertibleConv1x1
from src.model.FlowActNorms import ActNormHasLogdetRev, ActNormHasLogdet
from src.model.FlowAffineCouplingsAblation import CondAffineSeparatedAndCondFor, CondAffineSeparatedAndCondRev


class FlowStepFor(nn.Cell):
    """
    The train part of Flowstep Layer
    which has not Affine Layer
    """
    def __init__(self, in_channels):

        super().__init__()

        self.actnorm = ActNormHasLogdet(in_channels)
        self.invconv = InvertibleConv1x1(in_channels)

    def construct(self, x, logdet=None, rrdbResults=None):
        return self.normal_flow(x, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        z, logdet = self.actnorm(z, logdet=logdet)
        z, logdet = self.invconv(x=z, logdet=logdet)
        return z, logdet


class FlowStepRev(nn.Cell):
    """
    The test part of Flowstep Layer
    which has not Affine Layer
    """
    def __init__(self, in_channels):

        super().__init__()
        self.actnorm = ActNormHasLogdetRev(in_channels)
        self.invconv = InvertibleConv1x1(in_channels)

    def construct(self, x, logdet=None, rrdbResults=None):
        return self.reverse_flow(x, logdet, rrdbResults)

    def reverse_flow(self, z, logdet, rrdbResults=None):
        z, logdet = self.invconv(x=z, logdet=logdet)
        z, logdet = self.actnorm(z, logdet=logdet)
        return z, logdet


class FlowStepCondFor(nn.Cell):
    """
    The train part of Flowstep Layer
    which has Affine Layer
    """
    def __init__(self, in_channels, flow_coupling="additive"):

        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNormHasLogdet(in_channels)
        self.invconv = InvertibleConv1x1(in_channels)
        self.affine = CondAffineSeparatedAndCondFor(in_channels=in_channels)

    def construct(self, x, logdet=None, rrdbResults=None):
        return self.normal_flow(x, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        img_ft = rrdbResults
        z, logdet = self.actnorm(z, logdet=logdet)
        z, logdet = self.invconv(x=z, logdet=logdet)
        z, logdet = self.affine(x=z, logdet=logdet, ft=img_ft)
        return z, logdet


class FlowStepCondRev(nn.Cell):
    """
    The test part of Flowstep Layer
    which has Affine Layer
    """
    def __init__(self, in_channels):

        super().__init__()

        self.actnorm = ActNormHasLogdetRev(in_channels)
        self.invconv = InvertibleConv1x1(in_channels)
        self.affine = CondAffineSeparatedAndCondRev(in_channels=in_channels)

    def construct(self, x, logdet=None, rrdbResults=None):
        return self.reverse_flow(x, logdet, rrdbResults)

    def reverse_flow(self, z, logdet, rrdbResults=None):
        img_ft = rrdbResults
        z, logdet = self.affine(x=z, logdet=logdet, ft=img_ft)
        z, logdet = self.invconv(x=z, logdet=logdet)
        z, logdet = self.actnorm(z, logdet=logdet)
        return z, logdet
