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

"""Structure of Discriminator"""

import mindspore.nn as nn
import mindspore.ops.operations as P
from src.util.util import init_weights_network


class Discriminator(nn.Cell):
    """Structure of Discriminator"""

    def __init__(self, num_in_ch=3, num_feat=64):
        super(Discriminator, self).__init__()

        self.conv0_0 = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, pad_mode='pad',
                                 padding=1, has_bias=True)
        self.conv0_1 = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=4, stride=2, pad_mode='pad',
                                 padding=1, has_bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_features=num_feat, momentum=0.9, affine=True)

        self.conv1_0 = nn.Conv2d(in_channels=num_feat, out_channels=num_feat * 2, kernel_size=3, stride=1,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_features=num_feat * 2, momentum=0.9, affine=True)
        self.conv1_1 = nn.Conv2d(in_channels=num_feat * 2, out_channels=num_feat * 2, kernel_size=4, stride=2,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_features=num_feat * 2, momentum=0.9, affine=True)

        self.conv2_0 = nn.Conv2d(in_channels=num_feat * 2, out_channels=num_feat * 4, kernel_size=3, stride=1,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_features=num_feat * 4, momentum=0.9, affine=True)
        self.conv2_1 = nn.Conv2d(in_channels=num_feat * 4, out_channels=num_feat * 4, kernel_size=4, stride=2,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_features=num_feat * 4, momentum=0.9, affine=True)

        self.conv3_0 = nn.Conv2d(in_channels=num_feat * 4, out_channels=num_feat * 8, kernel_size=3, stride=1,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_features=num_feat * 8, momentum=0.9, affine=True)
        self.conv3_1 = nn.Conv2d(in_channels=num_feat * 8, out_channels=num_feat * 8, kernel_size=4, stride=2,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_features=num_feat * 8, momentum=0.9, affine=True)

        self.conv4_0 = nn.Conv2d(in_channels=num_feat * 8, out_channels=num_feat * 8, kernel_size=3, stride=1,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_features=num_feat * 8, momentum=0.9, affine=True)
        self.conv4_1 = nn.Conv2d(in_channels=num_feat * 8, out_channels=num_feat * 8, kernel_size=4, stride=2,
                                 pad_mode='pad', padding=1, has_bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_features=num_feat * 8, momentum=0.9, affine=True)

        self.linear1 = nn.Dense(in_channels=num_feat * 8 * 4 * 4, out_channels=100)
        self.linear2 = nn.Dense(in_channels=100, out_channels=1)

        # activation function
        self.lrelu = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):
        """discriminator compute graph
        Args:
            x(Tensor): low resolution image
        Outputs:
            Tensor
        """
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = P.Reshape()(feat, (P.Shape()(feat)[0], -1,))
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


def get_discriminator(num_in_ch):
    """Return discriminator."""
    net = Discriminator(num_in_ch=num_in_ch, num_feat=64)
    init_weights_network(net, 'normal', init_gain=0.02)
    return net
