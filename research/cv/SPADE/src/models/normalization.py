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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" SPADE Component """

import re
from mindspore import Tensor, nn
import mindspore as ms
import mindspore.ops as ops
from src.models.init_Parameter import XavierNormal
from src.util.instancenorm import InstanceNorm2d

class SPADE(nn.Cell):
    def __init__(self, config_text, norm_nc, label_nc, distribute):
        super(SPADE, self).__init__()
        assert config_text.startswith('spade')
        parsed = re.search(r'spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if param_free_norm_type == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=True)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        nhidden = 128

        pw = ks // 2
        xaviernormal = XavierNormal(0.02)
        weight_mlp_shared = xaviernormal.initialize([nhidden, label_nc, ks, ks])
        self.mlp_shared = nn.SequentialCell(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw, pad_mode='pad', \
                      has_bias=True, weight_init=Tensor(weight_mlp_shared, ms.float32), bias_init="zeros"),
            nn.ReLU()
        )
        weight_mlp_gamma = xaviernormal.initialize([norm_nc, nhidden, ks, ks])
        weight_mlp_beta = xaviernormal.initialize([norm_nc, nhidden, ks, ks])
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, pad_mode='pad', padding=pw, \
                                   has_bias=True, weight_init=Tensor(weight_mlp_gamma, ms.float32), bias_init="zeros")
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, pad_mode='pad', padding=pw, has_bias=True, \
                                  weight_init=Tensor(weight_mlp_beta, ms.float32), bias_init="zeros")

    def construct(self, x, segmap):
        normalized = self.param_free_norm(x)
        h = x.shape[2]
        w = x.shape[3]
        segmap = ops.ResizeNearestNeighbor((h, w))(segmap)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out
