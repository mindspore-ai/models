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

"""dbpns network."""

import mindspore
from mindspore import nn

from src.model.base_network import ConvBlock, UpBlock, DownBlock


class Net(nn.Cell):
    """Structure of dbpns network"""

    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        kernel = 8
        stride = 4
        padding = 2
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 1, 1, 0, activation=None, norm=None)

    def construct(self, x):
        """dbpns compute graph
        Args:
            x(Tensor): low resolution image
        Outputs:
            Tensor
        """
        x = self.feat0(x)
        x = self.feat1(x)
        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        op = mindspore.ops.Concat(1)
        x = self.output_conv(op((h2, h1)))
        return x
