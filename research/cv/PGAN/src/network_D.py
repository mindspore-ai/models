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
"""Dnet define"""
from mindspore import context
from mindspore import dtype as mstype
from mindspore import ops, nn

from src.customer_layer import EqualizedConv2d, EqualizedLinear, num_flat_features


class CustomMaxPool2x2(nn.Cell):
    """CustomMaxPool2x2"""

    def __init__(self):
        super().__init__()
        self.kernel_size = 2
        self.stride = 2
        self.max_op = ops.ReduceMax()

    def construct(self, x):
        """CustomMaxPool2x2

        Returns:
            output.
        """
        shape = x.shape
        batch, channel, height, width = shape
        x_reshaped = x.reshape((batch, channel, height // 2, 2, width // 2, 2))
        x_pooled = self.max_op(x_reshaped, 5)
        x_pooled = self.max_op(x_pooled, 3)
        return x_pooled


class DNet4_4_Train(nn.Cell):
    """DNet4_4_Train"""

    def __init__(self,
                 depthScale0,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 dimInput=3):
        super(DNet4_4_Train, self).__init__()
        self.dimInput = dimInput
        self.depthScale0 = depthScale0
        self.fromRGBLayers = EqualizedConv2d(dimInput, depthScale0, 1, padding=0, pad_mode="same", has_bias=True)
        self.dimEntryScale0 = depthScale0
        self.groupScale0 = EqualizedConv2d(self.dimEntryScale0, depthScale0, 3, padding=1, pad_mode="pad",
                                           has_bias=True)
        self.groupScale1 = EqualizedLinear(depthScale0 * 16, depthScale0)
        self.decisionLayer = EqualizedLinear(depthScale0, sizeDecisionLayer)
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        self.leakyReluLeak = leakyReluLeak
        self.cast = ops.Cast()
        self.scale = 4

    def construct(self, x, alpha=0.0):
        """DNet4_4_Train

        Returns:
            output.
        """
        x = self.leakyRelu(self.fromRGBLayers(x))
        x = self.leakyRelu(self.groupScale0(x))
        x = ops.reshape(x, (-1, num_flat_features(x)))
        x = self.leakyRelu(self.groupScale1(x))
        out = self.decisionLayer(x)
        return out


class DNet4_4_Last(nn.Cell):
    """DNet4_4_Last"""

    def __init__(self, dNet4_4):
        super(DNet4_4_Last, self).__init__()
        self.dimInput = dNet4_4.dimInput
        self.depthScale0 = dNet4_4.depthScale0
        self.fromRGBLayers = EqualizedConv2d(self.dimInput, self.depthScale0, 1, padding=0, pad_mode="same",
                                             has_bias=True)
        self.dimEntryScale0 = dNet4_4.depthScale0
        self.groupScale0 = EqualizedConv2d(self.dimEntryScale0, self.depthScale0, 3, padding=1, pad_mode="pad",
                                           has_bias=True)
        self.groupScale1 = EqualizedLinear(self.depthScale0 * 16, self.depthScale0)
        self.decisionLayer = dNet4_4.decisionLayer
        self.leakyRelu = nn.LeakyReLU(dNet4_4.leakyReluLeak)
        self.scale = 4

    def construct(self, x):
        """DNet4_4_Last

        Returns:
            output.
        """
        x = self.leakyRelu(self.groupScale0(x))
        x = ops.reshape(x, (-1, num_flat_features(x)))
        x = self.leakyRelu(self.groupScale1(x))
        out = self.decisionLayer(x)
        return out


class DNetNext_Train(nn.Cell):
    """DNetNext_Train"""

    def __init__(self,
                 depthScale0,
                 last_Dnet,
                 leakyReluLeak=0.2,
                 dimInput=3):
        super(DNetNext_Train, self).__init__()
        self.dimInput = dimInput
        self.fromRGBLayers = EqualizedConv2d(dimInput, depthScale0, 1, padding=0, pad_mode="same", has_bias=True)
        depthNewScale = depthScale0
        depthLastScale = last_Dnet.dimEntryScale0
        self.dimEntryScale0 = depthNewScale
        self.last_Dnet = last_Dnet
        self.scale = last_Dnet.scale * 2
        self.last_fromRGBLayers = last_Dnet.fromRGBLayers
        self.groupScale0 = EqualizedConv2d(depthNewScale, depthNewScale, 3, padding=1, pad_mode="pad", has_bias=True)
        self.groupScale1 = EqualizedConv2d(depthNewScale, depthLastScale, 3, padding=1, pad_mode="pad", has_bias=True)
        self.leakyRelu = nn.LeakyReLU(leakyReluLeak)
        if context.get_context('device_target') != "GPU":
            self.avgPool2d = ops.MaxPool(kernel_size=2, strides=2)
        else:
            # GPU does not support the second derivative for MaxPool (MaxPoolGradGrad).
            # Thus, we use our custom implementation based on the transpose and ReduceMax operations.
            self.avgPool2d = CustomMaxPool2x2()
        self.cast = ops.Cast()

    def construct(self, x, alpha=0):
        """DNetNext_Train

        Returns:
            output.
        """
        mid = self.cast(x, mstype.float16)
        y = self.avgPool2d(mid)
        y = self.cast(y, mstype.float32)
        y = self.leakyRelu(self.last_fromRGBLayers(y))
        x = self.leakyRelu(self.fromRGBLayers(x))
        x = self.leakyRelu(self.groupScale0(x))
        x = self.leakyRelu(self.groupScale1(x))
        x = self.cast(x, mstype.float16)
        x = self.avgPool2d(x)
        x = alpha * y + (1 - alpha) * x
        out = self.last_Dnet(x)
        return out


class DNetNext_Last(nn.Cell):
    """DNetNext_Last"""

    def __init__(self, dNetNext_Train):
        super(DNetNext_Last, self).__init__()
        self.scale = dNetNext_Train.scale
        self.dimInput = dNetNext_Train.dimInput
        self.fromRGBLayers = dNetNext_Train.fromRGBLayers
        self.dimEntryScale0 = dNetNext_Train.dimEntryScale0
        self.last_Dnet = dNetNext_Train.last_Dnet
        self.groupScale0 = dNetNext_Train.groupScale0
        self.groupScale1 = dNetNext_Train.groupScale1
        self.leakyRelu = dNetNext_Train.leakyRelu
        if context.get_context('device_target') != "GPU":
            self.avgPool2d = ops.MaxPool(kernel_size=2, strides=2)
        else:
            # GPU does not support the second derivative for MaxPool (MaxPoolGradGrad).
            # Thus, we use our custom implementation based on the transpose and ReduceMax operations.
            self.avgPool2d = CustomMaxPool2x2()
        self.cast = ops.Cast()

    def construct(self, x):
        """DNetNext_Last

        Returns:
            output.
        """
        x = self.leakyRelu(self.groupScale0(x))
        x = self.leakyRelu(self.groupScale1(x))
        x = self.cast(x, mstype.float16)
        x = self.avgPool2d(x)
        x = self.cast(x, mstype.float32)
        out = self.last_Dnet(x)
        return out
