#!/bin/bash
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

import mindspore.nn as nn
from src.Utils import L2_norm


class Lower(nn.Cell):

    def __init__(self):
        super(Lower, self).__init__()
        self.lower_opt = nn.SequentialCell(
            nn.Conv2d(in_channels=32,
                      out_channels=2,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=2, momentum=0.9, affine=False),
            nn.AvgPool2d(2, stride=2),
        )

        self.lower_sar = nn.SequentialCell(
            nn.Conv2d(in_channels=32,
                      out_channels=2,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=2, momentum=0.9, affine=False),
            nn.AvgPool2d(2, stride=2),
        )
        self.L2_norm = L2_norm()

    def construct(self, opt_feat, sar_feat):
        opt_lowers = self.lower_opt(opt_feat)
        sar_lowers = self.lower_sar(sar_feat)
        opt_lower = opt_lowers.view(opt_lowers.shape[0], -1)
        sar_lower = sar_lowers.view(sar_lowers.shape[0], -1)

        return self.L2_norm(opt_lower), opt_lowers, self.L2_norm(sar_lower), sar_lowers


class Decoder(nn.Cell):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.SequentialCell(
            nn.Conv2dTranspose(in_channels=2,
                               out_channels=16,
                               kernel_size=2,
                               stride=2,
                               pad_mode="pad",
                               has_bias=True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=16, momentum=0.9, affine=False),
            nn.ReLU(),

            nn.Conv2dTranspose(in_channels=16,
                               out_channels=16,
                               kernel_size=2,
                               stride=2,
                               pad_mode="pad",
                               has_bias=True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=16, momentum=0.9, affine=False),
            nn.ReLU(),

            nn.Conv2dTranspose(in_channels=16,
                               out_channels=1,
                               kernel_size=2,
                               stride=2,
                               pad_mode="pad",
                               has_bias=True),
            nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=1, momentum=0.9, affine=False),
            nn.Tanh(),
        )

    def construct(self, lower):
        img = self.decoder(lower)
        return img
