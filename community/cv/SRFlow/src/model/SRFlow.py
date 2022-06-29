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
The top layer of SRFlow
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from src.model.SRFlowNet import SRFlowNetFor, SRFlowNetRev


class SRFlowNetNllFor(nn.Cell):
    """
    The train part of SRFlow
    """
    def __init__(self, opt=None):
        super().__init__()
        opt_net = opt['network_G']
        self.SRFlow = SRFlowNetFor(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                   nf=opt_net['nf'], nb=opt_net['nb'],
                                   scale=opt['scale'], opt=opt)
        self.min_value = Tensor(0, mindspore.float32)
        self.max_value = Tensor(1, mindspore.float32)

    def construct(self, hr=None, lr=None):
        nll = self.SRFlow(gt=hr, lr=lr)
        reduce_mean = ops.ReduceMean(keep_dims=False)
        nll_loss = reduce_mean(nll)
        return nll_loss


class SRFlowNetNllRev(nn.Cell):
    """
    The test part of SRFlow
    """
    def __init__(self, opt=None):
        super().__init__()
        opt_net = opt['network_G']
        self.SRFlow = SRFlowNetRev(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                   nf=opt_net['nf'], nb=opt_net['nb'],
                                   scale=opt['scale'], opt=opt)
        self.min_value = Tensor(0, mindspore.float32)
        self.max_value = Tensor(1, mindspore.float32)

    def construct(self, hr=None, lr=None):
        sr = self.SRFlow(lr=lr)
        sr = ops.clip_by_value(sr, self.min_value, self.max_value)
        psnr = nn.PSNR()
        ssim = nn.SSIM()
        mean_psnr = psnr(sr, hr)
        mean_ssim = ssim(sr, hr)

        return sr, mean_psnr, mean_ssim
