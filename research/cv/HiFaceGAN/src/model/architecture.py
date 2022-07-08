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
"""Auxiliary classes for HiFaceGAN architecture"""
import numpy as np
import numpy.linalg as linalg
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

from src.model.initializer import XavierNormal


class Conv2dNormalized(nn.Cell):
    """Conv2d layer with spectral normalization"""

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, has_bias=False, pad=0, pad_mode="same"):
        super().__init__()
        self.conv2d = ops.Conv2D(out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                                 mode=1, pad=pad, pad_mode=pad_mode)
        self.bias_add = ops.BiasAdd(data_format="NCHW")
        self.has_bias = has_bias

        if self.has_bias:
            self.bias = Parameter(initializer('zeros', (out_channel,)), name='bias')

        self.weight_orig = Parameter(
            initializer(XavierNormal(gain=0.02), (out_channel, in_channel, kernel_size, kernel_size)),
            name='weight_orig'
        )

        self.weight_u = Parameter(self.initialize_param(out_channel, 1), requires_grad=False, name='weight_u')
        self.weight_v = Parameter(self.initialize_param(in_channel * kernel_size * kernel_size, 1), requires_grad=False,
                                  name='weight_v')

    @staticmethod
    def initialize_param(*param_shape):
        param = np.random.randn(*param_shape)
        return param / linalg.norm(param)

    def normalize_weights(self, weight_orig, u, v):
        """Weights normalization"""
        size = weight_orig.shape
        weight_mat = weight_orig.ravel().view(size[0], -1)

        if self.training:
            v = ops.matmul(weight_mat.T, u)
            v_norm = nn.Norm()(v)
            v = v / v_norm

            u = ops.matmul(weight_mat, v)
            u_norm = nn.Norm()(u)
            u = u / u_norm

            u = ops.depend(u, ops.assign(self.weight_u, u))
            v = ops.depend(v, ops.assign(self.weight_v, v))

            u = ops.stop_gradient(u)
            v = ops.stop_gradient(v)

        weight_norm = ops.matmul(u.T, ops.matmul(weight_mat, v))
        weight_sn = weight_mat / weight_norm
        weight_sn = weight_sn.view(*size)

        return weight_sn

    def construct(self, x):
        """Feed forward"""
        weight = self.normalize_weights(self.weight_orig, self.weight_u, self.weight_v)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class SPADE(nn.Cell):
    """Class for spatial adaptive denormalization (SPADE)"""

    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.norm = nn.BatchNorm2d(norm_nc, affine=False)

        if norm_nc > 128:
            self.n_hidden = 128
        else:
            self.n_hidden = norm_nc

        self.mlp_shared = nn.SequentialCell([
            nn.Conv2d(label_nc, self.n_hidden, kernel_size=3, pad_mode='same', has_bias=True),
            nn.ReLU()
        ])
        self.mlp_gamma = nn.Conv2d(self.n_hidden, norm_nc, kernel_size=3, pad_mode='same')
        self.mlp_beta = nn.Conv2d(self.n_hidden, norm_nc, kernel_size=3, pad_mode='same')

    def construct(self, x, segmap):
        """Feed forward"""
        normalized = self.norm(x)
        resize = ops.ResizeNearestNeighbor(x.shape[2:])
        segmap = resize(segmap)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * gamma + beta


class SPADEResnetBlock(nn.Cell):
    """SPADE ResNet block"""

    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # apply spectral norm if specified
        self.conv_0 = Conv2dNormalized(fin, fmiddle, kernel_size=3, pad_mode='same', has_bias=True)
        self.conv_1 = Conv2dNormalized(fmiddle, fout, kernel_size=3, pad_mode='same', has_bias=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dNormalized(fin, fout, kernel_size=1, pad_mode='valid')

        # define normalization layers
        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)

        self.actvn = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg):
        """Apply shortcut"""
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def construct(self, x, seg):
        """Feed forward"""
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        return x_s + dx
