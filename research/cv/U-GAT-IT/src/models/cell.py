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
"""
trainers
"""
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore import nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore

class Generator(nn.Cell):
    """
    Generator of CycleGAN, return fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.
    Args:
        genA2B (Cell): The generator network of domain A to domain B.
        genB2A (Cell): The generator network of domain B to domain A.
        use_identity (bool): Use identity loss or not. Default: True.
    Returns:
        Tensors, fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.
    Examples:
        >>> Generator(genA2B, genB2A)
    """
    def __init__(self, genA2B, genB2A):
        super(Generator, self).__init__()
        self.genA2B = genA2B
        self.genB2A = genB2A

    def construct(self, real_A, real_B):
        """construct"""
        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

        return fake_A2B, fake_A2B_cam_logit, fake_B2A, fake_B2A_cam_logit, fake_A2B2A, \
               fake_B2A2B, fake_A2A, fake_A2A_cam_logit, fake_B2B, fake_B2B_cam_logit

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, real_A, real_B):
        _, _, loss = self.network(real_A, real_B)
        return loss

grad_scale = C.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Reciprocal()(scale)

class TrainOneStepG(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        G (Cell): Generator with loss Cell. Note that loss function shouloss have been added.
        generator (Cell): Generator of CycleGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, G, generator, optimizer, sens=1.0, use_global_norm=False):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()

        self.G.disGA.set_grad(False)
        self.G.disGA.set_train(False)
        self.G.disGB.set_grad(False)
        self.G.disGB.set_train(False)
        self.G.disLA.set_grad(False)
        self.G.disLA.set_train(False)
        self.G.disLB.set_grad(False)
        self.G.disLB.set_train(False)

        self.use_global_norm = use_global_norm
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = mindspore.ParameterTuple(generator.trainable_params())
        self.net = WithLossCell(G)
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
        self.hyper_map = C.HyperMap()

    def construct(self, real_A, real_B):
        """construct"""
        weights = self.weights
        A2B, B2A, loss = self.G(real_A, real_B)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.net, weights)(real_A, real_B, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_array(self.sens)), grads)
            grads = C.clip_by_global_norm(grads)
        return A2B, B2A, ops.depend(loss, self.optimizer(grads))

class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of Cycle GAN discriminator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        G (Cell): Generator with loss Cell. Note that loss function shouloss have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, D, optimizer, sens=1.0, use_global_norm=False):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = mindspore.ParameterTuple(D.trainable_params())
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
        self.hyper_map = C.HyperMap()
        self.use_global_norm = use_global_norm


    def construct(self, real_A, real_B, fake_A2B, fake_B2A):
        """construct"""
        weights = self.weights
        loss = self.D(real_A, real_B, fake_A2B, fake_B2A)
        sens_d = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.D, weights)(real_A, real_B, fake_A2B, fake_B2A, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_array(self.sens)), grads)
            grads = C.clip_by_global_norm(grads)
        return ops.depend(loss, self.optimizer(grads))
