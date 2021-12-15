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
# ============================================================================s
"""Loss Computation of Generator and Discriminator"""
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore import nn, Tensor, ops

class GradientWithInput(nn.Cell):
    """Get Discriminator Gradient with Input"""

    def __init__(self, discriminator):
        super().__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discriminator = discriminator
        self.discriminator.set_train(mode=True)

    def construct(self, interpolates):
        decision_interpolate = self.discriminator(interpolates)

        return decision_interpolate


class WGANGPGradientPenalty(nn.Cell):
    """Define WGAN loss for SinGAN"""

    def __init__(self, discriminator, sens=1):
        super().__init__()
        self.gradient_op = ops.GradOperation(sens_param=True)

        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dim = ops.ReduceSum(keep_dims=True)

        self.sqrt = ops.Sqrt()
        self.discriminator = discriminator
        self.GradientWithInput = GradientWithInput(discriminator)
        self.uniformreal = ops.UniformReal()
        self.shape = (1, 1, 1, 1)
        self.sens = sens
    def construct(self, x_real, x_fake):
        """get gradient penalty"""
        alpha = self.uniformreal(self.shape)
        x_fake = ops.functional.stop_gradient(x_fake)
        x_hat = alpha * x_real +  (1 - alpha) * x_fake
        loss = self.discriminator(x_hat)
        sens = P.Fill()(loss.dtype, loss.shape, self.sens)
        gradient = self.gradient_op(self.GradientWithInput)(x_hat, sens)
        gradient = gradient / self.sens
        gradient_penalty = ops.ReduceMean()((nn.Norm(1)(gradient) - 1) ** 2)

        return gradient_penalty


class GenLoss(nn.Cell):
    """Define total Generator loss"""

    def __init__(self, args, generator, discriminator):
        super().__init__()
        self.net_G = generator
        self.net_D = discriminator

        self.lambda_rec = Tensor(args.alpha, mstype.float32)

        self.cyc_loss = P.ReduceMean()
        self.rec_loss = nn.MSELoss("mean")

    def construct(self, real, Z_opt, z_prev, noise, prev):
        """Get generator loss"""
        # rec loss
        x_rec = self.net_G(Z_opt, z_prev)
        G_rec_loss = self.lambda_rec * self.rec_loss(x_rec, real)

        # fake loss
        x_fake = self.net_G(noise, prev)
        fake_src = self.net_D(x_fake)
        G_fake_loss = -self.cyc_loss(fake_src)

        # g loss
        g_loss = G_fake_loss + G_rec_loss

        return (g_loss, G_fake_loss, G_rec_loss, x_fake, x_rec)


class DisLoss(nn.Cell):
    """Define total discriminator loss"""

    def __init__(self, args, generator, discriminator, sens=1):
        super().__init__()
        self.net_G = generator
        self.net_D = discriminator

        self.cyc_loss = P.ReduceMean()
        self.WGANLoss = WGANGPGradientPenalty(discriminator, sens)

        self.lambda_gp = Tensor(args.lambda_grad, mstype.float32)

    def construct(self, real, noise, prev):
        """Get discriminator loss"""
        # real loss
        real_src = self.net_D(real)
        D_real_loss = -self.cyc_loss(real_src)

        # fake loss
        x_fake = self.net_G(noise, prev)
        fake_src = self.net_D(ops.functional.stop_gradient(x_fake))
        D_fake_loss = self.cyc_loss(fake_src)

        # gp loss
        D_gp_loss = self.lambda_gp * self.WGANLoss(real, x_fake)

        # d loss
        d_loss = D_real_loss + D_fake_loss + D_gp_loss

        return (d_loss, D_real_loss, D_fake_loss, D_gp_loss)
