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
"""Cell Definition"""
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import nn, ops, Tensor
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)

class GenWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, real, Z_opt, z_prev, noise, prev):
        g_loss, _, _, _, _ = self.network(real, Z_opt, z_prev, noise, prev)
        return g_loss


class DisWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return discriminator loss
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, real, noise, prev):
        d_loss, _, _, _ = self.network(real, noise, prev)
        return d_loss


class TrainOneStepCellGen(nn.Cell):
    """Encapsulation class of AttGAN generator network training."""

    def __init__(self, generator, optimizer, sens=1.0, clip=10):
        super().__init__()
        self.optimizer = optimizer
        self.generator = generator
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.weights = optimizer.parameters
        self.network = GenWithLossCell(generator)
        self.network.add_flags(defer_inline=True)

        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()

        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)
        self.clip_value = Tensor(clip)
        self.clip_by_norm = nn.ClipByNorm()

    def construct(self, real, Z_opt, z_prev, noise, prev):
        """construct"""
        weights = self.weights
        loss, G_fake_loss, G_rec_loss, x_fake, x_rec = self.generator(real, Z_opt, z_prev, noise, prev)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(real, Z_opt, z_prev, noise, prev, sens)
        grads_cliped = ()
        for grad in grads:
            grad_cliped = self.clip_by_norm(grad, self.clip_value)
            grads_cliped = grads_cliped + (grad_cliped,)
        if self.reducer_flag:
            grads_cliped = self.grad_reducer(grads_cliped)
        return F.depend(loss, self.optimizer(grads_cliped)), G_fake_loss, G_rec_loss, x_fake, x_rec


class TrainOneStepCellDis(nn.Cell):
    """Encapsulation class of AttGAN discriminator network training."""

    def __init__(self, discriminator, optimizer, sens=1.0, clip=10):
        super().__init__()
        self.optimizer = optimizer
        self.discriminator = discriminator
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.weights = optimizer.parameters
        self.network = DisWithLossCell(discriminator)
        self.network.add_flags(defer_inline=True)

        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()

        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)
        self.clip_value = Tensor(clip)
        self.clip_by_norm = nn.ClipByNorm()

    def construct(self, real, noise, prev):
        """construct"""
        weights = self.weights
        loss, D_real_loss, D_fake_loss, D_gp_loss = self.discriminator(real, noise, prev)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(real, noise, prev, sens)
        grads_cliped = ()
        for grad in grads:
            grad_cliped = self.clip_by_norm(grad, self.clip_value)
            grads_cliped = grads_cliped + (grad_cliped,)
        if self.reducer_flag:
            grads_cliped = self.grad_reducer(grads_cliped)

        return F.depend(loss, self.optimizer(grads_cliped)), D_real_loss, D_fake_loss, D_gp_loss
