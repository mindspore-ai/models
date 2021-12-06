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

"""Structure of Generator"""

import mindspore.nn as nn
import mindspore.ops as ops
from src.util.util import init_weights_blocks


class ResidualDenseBlock(nn.Cell):
    """the structure of ResidualDenseBlock"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feat, out_channels=num_grow_ch, kernel_size=3, stride=1, pad_mode='pad',
                               padding=1, has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=num_feat + num_grow_ch, out_channels=num_grow_ch, kernel_size=3, stride=1,
                               pad_mode='pad', padding=1, has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=num_feat + 2 * num_grow_ch, out_channels=num_grow_ch, kernel_size=3,
                               stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=num_feat + 3 * num_grow_ch, out_channels=num_grow_ch, kernel_size=3,
                               stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.conv5 = nn.Conv2d(in_channels=num_feat + 4 * num_grow_ch, out_channels=num_feat, kernel_size=3, stride=1,
                               pad_mode='pad', padding=1, has_bias=True)

        self.lrelu = nn.LeakyReLU(alpha=0.2)
        self.cat = ops.Concat(1)
        # initialization
        init_weights_blocks([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], "kaiming", init_gain=0.1)

    def construct(self, x):
        """the ResidualDenseBlock compute graph"""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.cat((x, x1))))
        x3 = self.lrelu(self.conv3(self.cat((x, x1, x2))))
        x4 = self.lrelu(self.conv4(self.cat((x, x1, x2, x3))))
        x5 = self.conv5(self.cat((x, x1, x2, x3, x4)))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.SequentialCell(*layers)


class RRDB(nn.Cell):
    """the structure of RRDB"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def construct(self, x):
        """the RRDB block compute graph"""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Cell):
    """the structure of RRDBNet"""
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1,
                                    pad_mode='pad', padding=1, has_bias=True)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, pad_mode='pad',
                                   padding=1, has_bias=True)
        # upsample
        self.conv_up1 = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.conv_up2 = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, pad_mode='pad',
                                  padding=1, has_bias=True)
        self.conv_hr = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, pad_mode='pad',
                                 padding=1, has_bias=True)
        self.conv_last = nn.Conv2d(in_channels=num_feat, out_channels=num_out_ch, kernel_size=3, stride=1,
                                   pad_mode='pad', padding=1, has_bias=True)

        self.lrelu = nn.LeakyReLU(alpha=0.2)
        self.shape = ops.Shape()

    def construct(self, x):
        """the RRDBNet compute graph
        Args:
            x(Tensor): low resolution image
        Outputs:
            Tensor: high resolution image
        """
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        fea_size = self.shape(feat)
        feat = self.lrelu(self.conv_up1(ops.ResizeNearestNeighbor((fea_size[2] * 2, fea_size[3] * 2), True)(feat)))
        fea_size = self.shape(feat)
        feat = self.lrelu(self.conv_up2(ops.ResizeNearestNeighbor((fea_size[2] * 2, fea_size[3] * 2), True)(feat)))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def get_generator(num_in_ch, num_out_ch):
    """Return discriminator by args."""
    net = RRDBNet(num_in_ch=num_in_ch, num_out_ch=num_out_ch, num_feat=64, num_block=23, num_grow_ch=32)
    return net
