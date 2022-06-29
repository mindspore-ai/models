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
The Affine Layer of Glow
"""

import mindspore.nn as nn
import mindspore.ops as ops

from src.model.Flow import Conv2dZeros, Conv2d


class CondAffineSeparatedAndCondFor(nn.Cell):
    """
    The train part of Affine Layer
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_rrdb = 320
        self.kernel_hidden = 1
        self.n_hidden_layers = 1
        self.hidden_channels = 64
        self.affine_eps = 0.01
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def construct(self, x, logdet=None, ft=None):
        """
        construct
        """
        z = x
        concat = ops.Concat(axis=1)
        ft_fea = self.fFeatures(ft)
        scaleFt, shiftFt = self.feature_extract(ft_fea)
        z = (z + shiftFt) * scaleFt
        logdet = logdet + self.get_logdet(scaleFt)

        z1, z2 = self.split(z)
        ft_aff = concat((z1, ft))
        ft_aff = self.fAffine(ft_aff)
        scale, shift = self.feature_extract(ft_aff)
        z2 = (z2 + shift) * scale
        logdet = logdet + self.get_logdet(scale)

        z = concat((z1, z2))

        return z, logdet

    def get_logdet(self, scale):
        log = ops.Log()
        reduce_sum = ops.ReduceSum(keep_dims=False)
        logdet = reduce_sum(log(scale), [1, 2, 3])
        return logdet

    def feature_extract(self, ft):
        shift = ft[:, 0::2, ...]
        scale = ft[:, 1::2, ...]
        sigmoid = ops.Sigmoid()
        scale = (sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        """
        Create a SequentialCell
        """
        layers = []
        layers.append(Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                             pad_mode='pad', padding=1))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_hidden,
                                 stride=1, pad_mode='same', padding=0))
            layers.append(nn.ReLU())
        layers.append(Conv2dZeros(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  pad_mode='pad', padding=1))

        return nn.SequentialCell(*layers)


class CondAffineSeparatedAndCondRev(nn.Cell):
    """
    The test part of Affine Layer
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_rrdb = 320
        self.kernel_hidden = 1
        self.n_hidden_layers = 1
        self.hidden_channels = 64
        self.affine_eps = 0.01
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def construct(self, x, logdet=None, ft=None):
        """
        construct
        """
        z = x
        concat = ops.Concat(axis=1)
        z1, z2 = self.split(z)
        ft_aff = concat((z1, ft))
        ft_aff = self.fAffine(ft_aff)
        scale, shift = self.feature_extract(ft_aff)
        z2 = (z2 / scale) - shift
        z = concat((z1, z2))
        logdet = logdet - self.get_logdet(scale)

        ft_fea = self.fFeatures(ft)
        scaleFt, shiftFt = self.feature_extract(ft_fea)
        z = (z / scaleFt) - shiftFt
        logdet = logdet - self.get_logdet(scaleFt)

        return z, logdet

    def get_logdet(self, scale):
        log = ops.Log()
        reduce_sum = ops.ReduceSum(keep_dims=False)
        logdet = reduce_sum(log(scale), [1, 2, 3])
        return logdet

    def feature_extract(self, ft):
        shift = ft[:, 0::2, ...]
        scale = ft[:, 1::2, ...]
        sigmoid = ops.Sigmoid()
        scale = (sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        """
        Create a SequentialCell
        """
        layers = []
        layers.append(Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                             pad_mode='pad', padding=1))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_hidden,
                                 stride=1, pad_mode='same', padding=0))
            layers.append(nn.ReLU())
        layers.append(Conv2dZeros(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  pad_mode='pad', padding=1))

        return nn.SequentialCell(*layers)
