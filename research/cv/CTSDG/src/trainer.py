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
"""trainer"""

from mindspore import Parameter
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from src.losses import WithLossCell


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""
    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Parameter(Tensor(initial_scale_sense, dtype=mstype.float32), name="scale_sense")
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


class GTrainOneStepCell(TrainOneStepCell):
    """Generator TrainOneStepCell"""
    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super().__init__(network, optimizer, initial_scale_sense)
        self.network.vgg_feat_extractor.set_grad(False)
        self.network.vgg_feat_extractor.set_train(False)
        self.network.discriminator.set_grad(False)
        self.network.discriminator.set_train(False)

        self.net_grad = WithLossCell(network)

    def construct(self, *inputs):
        """construct"""
        loss, pred = self.network(*inputs)
        grads = self.grad(self.net_grad, self.weights)(*inputs, self.scale_sense * 1.)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        opt_res = self.optimizer(grads)
        return ops.depend(loss, opt_res), ops.depend(pred, opt_res)


class DTrainOneStepCell(TrainOneStepCell):
    """Discriminator TrainOneStepCell"""
    def construct(self, *inputs):
        """construct"""
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense * 1.)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class CTSDGTrainer:
    """CTSDGTrainer"""
    def __init__(self, train_one_step_g, train_one_step_d):
        super().__init__()
        self.train_one_step_g = train_one_step_g
        self.train_one_step_d = train_one_step_d

    def __call__(self, *inputs):
        ground_truth, _, edge, gray_image = inputs
        loss_g, output = self.train_one_step_g(*inputs)
        loss_d = self.train_one_step_d(ground_truth, gray_image, edge, output)
        return loss_g, loss_d
