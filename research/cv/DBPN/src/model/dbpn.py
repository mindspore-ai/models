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

"""dbpnv1 network."""

import mindspore
from mindspore import nn

from src.model.base_network import ConvBlock, UpBlock, DownBlock, DDownBlock, DUpBlock


class Net(nn.Cell):
    """Structure of dbpnv1 network"""
    def __init__(self, num_channels, base_filter, feat, num_stages,
                 scale_factor):
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
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu')
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu')
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = DDownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = DUpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = DDownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = DUpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = DDownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = DUpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = DDownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = DUpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = DDownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = DUpBlock(base_filter, kernel, stride, padding, 6)
        self.down7 = DDownBlock(base_filter, kernel, stride, padding, 7)
        self.up8 = DUpBlock(base_filter, kernel, stride, padding, 7)
        self.down8 = DDownBlock(base_filter, kernel, stride, padding, 8)
        self.up9 = DUpBlock(base_filter, kernel, stride, padding, 8)
        self.down9 = DDownBlock(base_filter, kernel, stride, padding, 9)
        self.up10 = DUpBlock(base_filter, kernel, stride, padding, 9)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1, 1, activation=None)
        self.op = mindspore.ops.Concat(1)

    def construct(self, x):
        """dbpn compute graph
        Args:
            x(Tensor): low resolution image
        Outputs:
            Tensor
        """
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = self.op((h2, h1))
        l = self.down2(concat_h)

        concat_l = self.op((l, l1))
        h = self.up3(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down3(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up4(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down4(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up5(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down5(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up6(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down6(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up7(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down7(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up8(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down8(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up9(concat_l)

        concat_h = self.op((h, concat_h))
        l = self.down9(concat_h)

        concat_l = self.op((l, concat_l))
        h = self.up10(concat_l)

        concat_h = self.op((h, concat_h))
        x = self.output_conv(concat_h)
        return x
