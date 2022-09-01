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
The Split Layer of Glow
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np

from src.model.Flow import Conv2dZeros


class Split2dFor(nn.Cell):
    """
    The train part of Split Layer
    """
    def __init__(self, num_channels, consume_ratio=0.5, opt=None):
        super().__init__()
        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume
        self.conv = Conv2dZeros(in_channels=self.num_channels_pass, out_channels=self.num_channels_consume * 2)
        self.opt = opt

    def split2d_prior(self, z):
        h = self.conv(z)
        return h[:, 0::2, ...], h[:, 1::2, ...]

    def construct(self, x, logdet=0., rrdbResults=None):
        z1, z2 = self.split_ratio(x)
        mean, logs = self.split2d_prior(z1)
        logdet = logdet + self.get_logdet(logs, mean, z2)
        return z1, logdet

    def get_logdet(self, logs, mean, z2):
        log2PI = np.log(2 * np.pi)
        exp = ops.Exp()
        reduce_sum = ops.ReduceSum(keep_dims=False)
        logdet_diff = -0.5 * (logs * 2. + ((z2 - mean) ** 2) / exp(logs * 2.) + log2PI)
        logdet_diff = reduce_sum(logdet_diff, (1, 2, 3))

        return logdet_diff

    def split_ratio(self, x):
        z1, z2 = x[:, :self.num_channels_pass, ...], x[:, self.num_channels_pass:, ...]
        return z1, z2


class Split2dRev(nn.Cell):
    """
    The test part of Split Layer
    """
    def __init__(self, num_channels, consume_ratio=0.5, opt=None):
        super().__init__()
        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume
        self.conv = Conv2dZeros(in_channels=self.num_channels_pass,
                                out_channels=self.num_channels_consume * 2)
        self.heat = opt['heat']

    def split2d_prior(self, z):
        h = self.conv(z)
        return h[:, 0::2, ...], h[:, 1::2, ...]

    def construct(self, x, logdet=0., rrdbResults=None):
        """
        construct
        """
        z1 = x
        mean, logs = self.split2d_prior(z1)
        zeros = ops.Zeros()
        ones = ops.Ones()
        shape = ops.Shape()
        exp = ops.Exp()
        mean_shape = shape(mean)
        eps = ops.normal(shape=mean_shape, mean=zeros(mean_shape, mindspore.float32),
                         stddev=ones(mean_shape, mindspore.float32) * self.heat, seed=1)
        z2 = mean + exp(logs) * eps
        concat = ops.Concat(axis=1)
        z = concat((z1, z2))
        logdet = logdet - self.get_logdet(logs, mean, z2)

        return z, logdet

    def get_logdet(self, logs, mean, z2):
        log2PI = np.log(2 * np.pi)
        exp = ops.Exp()
        reduce_sum = ops.ReduceSum(keep_dims=False)
        logdet_diff = -0.5 * (logs * 2. + ((z2 - mean) ** 2) / exp(logs * 2.) + log2PI)
        logdet_diff = reduce_sum(logdet_diff, (1, 2, 3))

        return logdet_diff
