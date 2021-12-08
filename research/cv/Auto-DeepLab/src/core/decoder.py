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
"""Decoder of Auto-DeepLab.
Here we reused the decoder architecture in DeepLabV3+ as our decoder in Auto-DeepLab
"""
import mindspore.nn as nn
import mindspore.ops as ops
from ..modules.bn import NormReLU


class Decoder(nn.Cell):
    """Decoder"""
    def __init__(self,
                 num_classes,
                 low_level_inplanes,
                 momentum,
                 eps,
                 parallel=True):
        super(Decoder, self).__init__()
        c_high = 256
        c_low = 48
        c_cat = c_high + c_low

        self.conv1 = nn.Conv2d(low_level_inplanes, c_low, 1, has_bias=False)
        self.bn1 = NormReLU(c_low, momentum, eps, parallel=parallel)

        self.interpolate = nn.ResizeBilinear()
        self.cat = ops.Concat(axis=1)
        self.last_conv = nn.SequentialCell(
            nn.Conv2d(c_cat, 256, 3, 1, pad_mode='same', has_bias=False, weight_init='HeNormal'),
            NormReLU(256, momentum, eps, parallel=parallel),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, 1, pad_mode='same', has_bias=False, weight_init='HeNormal'),
            NormReLU(256, momentum, eps, parallel=parallel),
            nn.Dropout(0.9),
            nn.Conv2d(256, num_classes, 1, 1, weight_init='HeNormal'))

    def construct(self, high_level_feat, low_level_feat):
        """construct"""
        low_level_feat_0 = self.conv1(low_level_feat)
        low_level_feat_1 = self.bn1(low_level_feat_0)

        high_level_feat_0 = self.interpolate(high_level_feat, (low_level_feat_1.shape[2:]), None, True)
        concat_feat = self.cat((high_level_feat_0, low_level_feat_1))
        output = self.last_conv(concat_feat)
        return output
