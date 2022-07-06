# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""cell define"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import OnesLike, ZerosLike


class GenWithLossCell(nn.Cell):
    """
    CGAN generator loss.
    Args:
        generator (Cell): Generator of GAN.
        discriminator (Cell): Discriminator of GAN.
    Outputs:
        the losses of generator.
    """
    def __init__(self, generator, discriminator):
        super(GenWithLossCell, self).__init__()

        self.dis_loss = nn.BCELoss(reduction="mean")
        self.generator = generator
        self.discriminator = discriminator

    def construct(self, noise, labels):
        """
        construct

        Args:
          noise(Tensor): noise
          labels(Tensor): labels

        Returns:
          img_loss:  img and loss
        """
        fake_img = self.generator(noise, labels)
        fake_out = self.discriminator(fake_img, labels)
        ones = OnesLike()(fake_out)
        loss_G = self.dis_loss(fake_out, ones)
        img_loss = (fake_img, loss_G)

        return img_loss

class DisWithLossCell(nn.Cell):
    """
    CGAN generator loss.
    Args:
        args (class): Option class.
        generator (Cell): Generator of GAN.
        discriminator (Cell): Discriminator of GAN.
    Outputs:
        the losses of discriminator.
    """
    def __init__(self, generator, discriminator):
        super(DisWithLossCell, self).__init__()

        self.dis_loss = nn.BCELoss(reduction="mean")
        self.generator = generator
        self.discriminator = discriminator

    def construct(self, real_img, noise, label):
        """
        construct

        Args:
            real_img(Tensor): real img
            noise(Tensor): noise
            label(Tensor): labels

        Returns:
            loss_D: loss
        """
        fake_img = self.generator(noise, label)
        fake_out = self.discriminator(fake_img, label)
        real_out = self.discriminator(real_img, label)
        zeros = ZerosLike()(fake_out)
        ones = OnesLike()(real_out)
        loss_D_f = self.dis_loss(fake_out, zeros)
        loss_D_r = self.dis_loss(real_out, ones)
        loss_D = (loss_D_f + loss_D_r) * 0.5
        return loss_D

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """

    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, noise, label):
        _, lg = self.network(noise, label)
        return lg

class TrainOneStepG(nn.Cell):
    """
    Encapsulation class of CGAN generator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        generator (Cell): Generator of CGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, G_with_loss, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G_with_loss = G_with_loss
        self.G_with_loss.set_grad()
        self.G_with_loss.set_train()

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.loss_net = WithLossCell(G_with_loss)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, noise, labels):
        """
        construct

        Args:
          noise(Tensor): noise
          labels(Tensor): labels

        Returns:
          fake_img: net output
          loss: loss
        """
        fake_img, loss_G = self.G_with_loss(noise, labels)
        sens = ops.Fill()(ops.DType()(loss_G), ops.Shape()(loss_G), self.sens)
        grads_g = self.grad(self.loss_net, self.weights)(noise, labels, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)
        loss = ops.depend(loss_G, self.optimizer(grads_g))
        return fake_img, loss


class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of CGAN discriminator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        D (Cell): D with loss Cell. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, D_with_loss, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.D_with_loss = D_with_loss
        self.D_with_loss.set_grad()
        self.D_with_loss.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, real_img, noise, labels):
        """
        construct

        Args:
            real_img
            noise(Tensor): noise
            labels(Tensor): labels

        Returns:
            loss: loss
        """
        weights = self.weights
        ld = self.D_with_loss(real_img, noise, labels)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.D_with_loss, weights)(real_img, noise, labels, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)
        loss = ops.depend(ld, self.optimizer(grads_d))
        return loss
