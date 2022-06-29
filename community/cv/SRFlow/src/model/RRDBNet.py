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
The ResNet of SRFlow
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeNormal


class ResidualDenseBlock_5C(nn.Cell):
    """
    The Residual Block
    """
    def __init__(self, nf=64, gc=32, has_bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, padding=1, has_bias=has_bias, pad_mode='pad')
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, padding=1, has_bias=has_bias, pad_mode='pad')
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, padding=1, has_bias=has_bias, pad_mode='pad')
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, padding=1, has_bias=has_bias, pad_mode='pad')
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, padding=1, has_bias=has_bias, pad_mode='pad')
        self.lrelu = nn.LeakyReLU(alpha=0.2)
        self.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def construct(self, x):
        concat = ops.Concat(axis=1)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(concat((x, x1))))
        x3 = self.lrelu(self.conv3(concat((x, x1, x2))))
        x4 = self.lrelu(self.conv4(concat((x, x1, x2, x3))))
        x5 = self.conv5(concat((x, x1, x2, x3, x4)))
        return x5 * 0.2 + x

    def initialize_weights(self, net_l, scale=1.0):
        """

        Args:
            net_l: net
            scale: scale

        Returns: init weights

        """
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.cells():
                shape = ops.Shape()
                m.weight = initializer(HeNormal(negative_slope=0, mode='fan_in'), shape(m.weight), mindspore.float32)
                m.weight.data *= scale
                if m.bias is not None:
                    zeroslike = ops.ZerosLike()
                    m.bias.data = zeroslike(m.bias.data)


class RRDB(nn.Cell):
    """
    The ResNet Block
    """
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def construct(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Cell):
    """
    The main part of ResNet
    """
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        super(RRDBNet, self).__init__()
        self.RRDB_trunk = nn.CellList()
        self.nb = nb
        for _ in range(self.nb):
            self.RRDB_trunk.append(RRDB(nf=nf, gc=gc))
        self.scale = scale
        self.blocks = self.opt['network_G']['flow']['stackRRDB']['blocks']

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, padding=1, has_bias=True, pad_mode='pad')
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True, pad_mode='pad')
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True, pad_mode='pad')
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True, pad_mode='pad')
        self.lrelu = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):
        """
        construct
        """
        fea = [i for i in range(0, self.nb + 1)]
        fea[0] = self.conv_first(x)
        for idx, layer in enumerate(self.RRDB_trunk):
            fea[idx+1] = layer(fea[idx])

        concat = ops.Concat(axis=1)
        out = concat((fea[2], fea[9]))
        out = concat((out, fea[16]))
        out = concat((out, fea[23]))

        feature = fea[self.nb]

        trunk = self.trunk_conv(feature)

        last_lr_fea = feature + trunk

        shape = ops.Shape()

        resize_nearest_neighbor = ops.ResizeNearestNeighbor((shape(last_lr_fea)[2] * 2, shape(last_lr_fea)[3] * 2))
        fea_up2 = resize_nearest_neighbor(last_lr_fea)
        fea_up2 = self.upconv1(fea_up2)

        feature = self.lrelu(fea_up2)

        resize_nearest_neighbor = ops.ResizeNearestNeighbor((shape(feature)[2] * 2, shape(feature)[3] * 2))
        fea_up4 = resize_nearest_neighbor(feature)
        fea_up4 = self.upconv2(fea_up4)

        resize_bilinear = nn.ResizeBilinear()
        fea_up0 = resize_bilinear(x=last_lr_fea, size=(shape(last_lr_fea)[2] // 2, shape(last_lr_fea)[3] // 2))

        keys = ['fea_up0', 'fea_up1', 'fea_up2', 'fea_up4']

        results = {
            'fea_up0': fea_up0,
            'fea_up1': last_lr_fea,
            'fea_up2': fea_up2 / 5,
            'fea_up4': fea_up4 / 5
        }

        for k in keys:
            h = shape(results[k])[2]
            w = shape(results[k])[3]
            resize_nearest_neighbor = ops.ResizeNearestNeighbor((h, w))
            results[k] = concat((results[k], resize_nearest_neighbor(out)))

        return results
