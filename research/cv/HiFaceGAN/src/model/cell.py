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
"""HiFaceGAN training network wrapper"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode


class WithLossCell(nn.Cell):
    """Wrap the network with loss function to return generator loss"""

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, lq, hq):
        """Construct forward graph"""
        output = self.network(lq, hq)
        return output[0]


class TrainOneStepD(nn.Cell):
    """Discriminator training package class"""

    def __init__(self, discriminator_with_loss_cell, optimizer):
        super().__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.discriminator = discriminator_with_loss_cell
        self.discriminator.set_grad()
        self.discriminator.set_train()
        self.grad = ops.GradOperation(get_by_list=True)
        self.weights = ms.ParameterTuple(discriminator_with_loss_cell.trainable_params())
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, lq, hq, generated):
        """Construct forward graph"""
        weights = self.weights
        loss, loss_fake, loss_real = self.discriminator(lq, hq, generated)
        grads_d = self.grad(self.discriminator, weights)(lq, hq, generated)
        if self.reducer_flag:
            grads_d = self.grad_reducer(grads_d)
        return ops.depend(loss, self.optimizer(grads_d)), loss_fake, loss_real


class TrainOneStepG(nn.Cell):
    """Generator training package class"""

    def __init__(self, generator_with_loss_cell, generator, optimizer):
        super().__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.generator_with_loss_cell = generator_with_loss_cell
        self.generator_with_loss_cell.set_grad()
        self.generator_with_loss_cell.set_train()
        self.generator_with_loss_cell.discriminator.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True)
        self.weights = ms.ParameterTuple(generator.trainable_params())
        self.net = WithLossCell(generator_with_loss_cell)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, lq, hq):
        """Construct forward graph"""
        weights = self.weights
        loss, vgg_loss, gan_loss, gan_feat_loss, generated = self.generator_with_loss_cell(lq, hq)
        grads_g = self.grad(self.net, weights)(lq, hq)
        if self.reducer_flag:
            grads_g = self.grad_reducer(grads_g)

        opt_res = self.optimizer(grads_g)
        return ops.depend(loss, opt_res), vgg_loss, gan_loss, gan_feat_loss, ops.depend(generated, opt_res)
