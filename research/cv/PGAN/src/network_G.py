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
"""Gnet define"""
from mindspore import ops, nn
import mindspore
from src.customer_layer import EqualizedConv2d, EqualizedLinear, num_flat_features


class NormalizationLayer(nn.Cell):
    """NormalizationLayer"""

    def __init__(self, epsilon=1e-8):
        super(NormalizationLayer, self).__init__()
        self.mean_op = ops.ReduceMean(keep_dims=True)
        self.rsqrt = ops.Rsqrt()
        self.epsilon = mindspore.Tensor(epsilon, mindspore.float32)

    def construct(self, x):
        """NormalizationLayer

        Returns:
            output.
        """
        y = self.mean_op(x ** 2, 1) + self.epsilon
        y = self.rsqrt(y)
        y = x * y
        return y


class GNet4_4_Train(nn.Cell):
    """GNet4_4_Train"""

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 leakyReluLeak=0.2,
                 dimOutput=3,
                 ):
        super(GNet4_4_Train, self).__init__()
        self.dimLatent = dimLatent
        self.depthScale = depthScale0
        self.formatLayer = EqualizedLinear(self.dimLatent, 16 * depthScale0)
        self.groupScale0 = EqualizedConv2d(depthScale0, depthScale0, 3, padding=1, pad_mode="pad", has_bias=True)
        self.toRGBLayers = EqualizedConv2d(depthScale0, dimOutput, 1, padding=0, pad_mode="same", has_bias=True)
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        self.normalizationLayer = NormalizationLayer()
        self.scale = 4

    def construct(self, x, alpha=0.0):
        """GNet4_4_Train

        Returns:
            output.
        """
        x = self.normalizationLayer(x)
        x = ops.reshape(x, (-1, num_flat_features(x)))
        x = self.leakyRelu(self.formatLayer(x))
        x = ops.reshape(x, (x.shape[0], -1, 4, 4))
        x = self.normalizationLayer(x)
        x = self.leakyRelu(self.groupScale0(x))
        x = self.normalizationLayer(x)
        x = self.toRGBLayers(x)
        return x


class GNet4_4_last(nn.Cell):
    """GNet4_4_last"""

    def __init__(self, gNet4_4_Train):
        super(GNet4_4_last, self).__init__()
        self.dimLatent = gNet4_4_Train.dimLatent
        self.depthScale = gNet4_4_Train.depthScale
        self.formatLayer = gNet4_4_Train.formatLayer
        self.groupScale0 = gNet4_4_Train.groupScale0
        self.toRGBLayers = gNet4_4_Train.toRGBLayers
        self.leakyRelu = gNet4_4_Train.leakyRelu
        self.normalizationLayer = gNet4_4_Train.normalizationLayer
        self.scale = 4

    def construct(self, x):
        """GNet4_4_last

        Returns:
            output.
        """
        x = self.normalizationLayer(x)
        x = ops.reshape(x, (-1, num_flat_features(x)))
        x = self.leakyRelu(self.formatLayer(x))
        x = ops.reshape(x, (x.shape[0], -1, 4, 4))
        x = self.normalizationLayer(x)
        x = self.leakyRelu(self.groupScale0(x))
        x = self.normalizationLayer(x)
        return x


class GNetNext_Train(nn.Cell):
    """GNetNext_Train"""

    def __init__(self,
                 depthScale0,
                 last_Gnet,
                 leakyReluLeak=0.2,
                 normalization=True,
                 dimOutput=3,
                 ):
        super(GNetNext_Train, self).__init__()
        self.depthScale = depthScale0
        self.last_Gnet = last_Gnet
        self.last_toRGBLayers = last_Gnet.toRGBLayers
        depthLastScale = self.last_Gnet.depthScale
        self.groupScale0 = EqualizedConv2d(depthLastScale, depthScale0, 3, padding=1, pad_mode="pad", has_bias=True)
        self.groupScale1 = EqualizedConv2d(depthScale0, depthScale0, 3, padding=1, pad_mode="pad", has_bias=True)
        self.toRGBLayers = EqualizedConv2d(depthScale0, dimOutput, 1, padding=0, pad_mode="same", has_bias=True)
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        self.normalizationLayer = NormalizationLayer()
        self.scale = self.last_Gnet.scale * 2
        self.Upscale2d = ops.ResizeNearestNeighbor((self.scale, self.scale))

    def construct(self, x, alpha=0):
        """GNetNext_Train

        Returns:
            output.
        """
        x = self.last_Gnet(x)
        y = self.last_toRGBLayers(x)
        y = self.Upscale2d(y)
        x = self.Upscale2d(x)
        x = self.leakyRelu(self.groupScale0(x))
        x = self.normalizationLayer(x)
        x = self.leakyRelu(self.groupScale1(x))
        x = self.normalizationLayer(x)
        x = self.leakyRelu(self.toRGBLayers(x))
        x = alpha * y + (1.0 - alpha) * x
        return x


class GNetNext_Last(nn.Cell):
    """GNetNext_Last"""

    def __init__(self, gNetNext_Train):
        super(GNetNext_Last, self).__init__()
        self.depthScale = gNetNext_Train.depthScale
        self.last_Gnet = gNetNext_Train.last_Gnet
        self.groupScale0 = gNetNext_Train.groupScale0
        self.groupScale1 = gNetNext_Train.groupScale1
        self.toRGBLayers = gNetNext_Train.toRGBLayers
        self.leakyRelu = gNetNext_Train.leakyRelu
        self.normalizationLayer = gNetNext_Train.normalizationLayer
        self.scale = gNetNext_Train.scale
        self.Upscale2d = gNetNext_Train.Upscale2d

    def construct(self, x):
        """GNetNext_Last

        Returns:
            output.
        """
        x = self.last_Gnet(x)
        x = self.Upscale2d(x)
        x = self.leakyRelu(self.groupScale0(x))
        x = self.normalizationLayer(x)
        x = self.leakyRelu(self.groupScale1(x))
        x = self.normalizationLayer(x)
        return x
