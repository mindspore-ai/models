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
"""model."""
import mindspore.nn as nn


class REDNet30(nn.Cell):
    """model"""
    def __init__(self, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()

        self.num_layers = num_layers
        conv_layers = []
        deconv_layers = []
        conv_layers.append(nn.Conv2d(3, num_features, kernel_size=3, stride=1,
                                     pad_mode="pad", padding=1, has_bias=False,
                                     weight_init="XavierUniform"))

        for _ in range(num_layers - 1):
            conv_layers.append(nn.Conv2d(num_features, num_features, kernel_size=3,
                                         pad_mode="pad", padding=1, has_bias=False,
                                         weight_init="XavierUniform"))

        for _ in range(num_layers - 1):
            deconv_layers.append(nn.Conv2dTranspose(num_features, num_features, kernel_size=3,
                                                    pad_mode="pad", padding=1, has_bias=False,
                                                    weight_init="XavierUniform"))
        deconv_layers.append(nn.Conv2dTranspose(num_features, 3, kernel_size=3, stride=1,
                                                pad_mode="same", has_bias=False,
                                                weight_init="XavierUniform"))

        self.conv_layers = nn.CellList(conv_layers)
        self.deconv_layers = nn.CellList(deconv_layers)
        self.relu = nn.ReLU()

    def construct(self, x):
        """model"""
        residual = x
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = self.relu(x)

            if (i + 1) % 2 == 0 and len(conv_feats) < self.num_layers//2:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if i != 14:
                x = self.relu(x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)
        return x
