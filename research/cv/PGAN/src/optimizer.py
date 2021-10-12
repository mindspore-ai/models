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
from mindspore import ops
from mindspore import nn
from mindspore.ops import constexpr
import mindspore
import numpy as np



@constexpr
def generate_tensor(batch_size):
    """generate_tensor

    Returns:
        output.
    """
    np_array = np.random.randn(batch_size, 1, 1, 1)
    return mindspore.Tensor(np_array, mindspore.float32)


class GradientWithInput(nn.Cell):
    """GradientWithInput"""

    def __init__(self, discrimator):
        super(GradientWithInput, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discrimator = discrimator

    def construct(self, interpolates, alpha):
        """GradientWithInput

        Returns:
            output.
        """
        decisionInterpolate = self.discrimator(interpolates, alpha)
        decisionInterpolate = self.reduce_sum(decisionInterpolate, 0)
        return decisionInterpolate


class WGANGPGradientPenalty(nn.Cell):
    """WGANGPGradientPenalty"""

    def __init__(self, discrimator, lambdaGP=10):
        super(WGANGPGradientPenalty, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dim = ops.ReduceSum(keep_dims=True)
        self.sqrt = ops.Sqrt()
        self.discrimator = discrimator
        self.gradientWithInput = GradientWithInput(discrimator)
        self.lambdaGP = mindspore.Tensor(lambdaGP, mindspore.float32)
        self.gradient_op = ops.GradOperation()

    def construct(self, input_x, fake, input_alpha):
        """WGANGPGradientPenalty

        Returns:
            output.
        """
        batch_size = input_x.shape[0]
        alpha = generate_tensor(batch_size)
        alpha = alpha.expand_as(input_x)
        interpolates = alpha * input_x + ((1 - alpha) * fake)
        gradient = self.gradient_op(self.gradientWithInput)(interpolates, input_alpha)
        gradient = ops.reshape(gradient, (batch_size, -1))
        gradient = self.sqrt(self.reduce_sum(gradient * gradient, 1))
        gradient_penalty = self.reduce_sum_keep_dim((gradient - 1.0) ** 2) * self.lambdaGP
        return gradient_penalty


class AllLossD(nn.Cell):
    """AllLossD"""

    def __init__(self, netD):
        super(AllLossD, self).__init__()
        self.netD = netD
        self.wGANGPGradientPenalty = WGANGPGradientPenalty(self.netD)
        self.reduce_sum = ops.ReduceSum()
        self.epsilonLoss = EpsilonLoss(0.001)
        self.scalr_summary = ops.ScalarSummary()
        self.summary = ops.TensorSummary()

    def construct(self, real, fake, alpha):
        """AllLossD

        Returns:
            output.
        """
        predict_real = self.netD(real, alpha)
        loss_real = -self.reduce_sum(predict_real, 0)
        predict_fake = self.netD(fake, alpha)
        loss_fake = self.reduce_sum(predict_fake, 0)
        lossD_Epsilon = self.epsilonLoss(predict_real)
        lossD_Grad = self.wGANGPGradientPenalty(real, fake, alpha)
        all_loss = loss_real + loss_fake + lossD_Grad + lossD_Epsilon
        return all_loss


class AllLossG(nn.Cell):
    """AllLossG"""

    def __init__(self, netG, netD):
        super(AllLossG, self).__init__()
        self.netG = netG
        self.netD = netD
        self.reduce_sum = ops.ReduceSum()

    def construct(self, inputNoise, alpha):
        """AllLossG

        Returns:
            output.
        """
        fake = self.netG(inputNoise, alpha)
        predict_fake = self.netD(fake, alpha)
        loss_fake = -self.reduce_sum(predict_fake, 0)
        return loss_fake


class EpsilonLoss(nn.Cell):
    """EpsilonLoss"""

    def __init__(self, epsilonD):
        super(EpsilonLoss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.epsilonD = mindspore.Tensor(epsilonD, mindspore.float32)

    def construct(self, predRealD):
        """EpsilonLoss

        Returns:
            output.
        """
        return self.reduce_sum(predRealD ** 2) * self.epsilonD
