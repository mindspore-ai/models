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
"""Train cell for DeepID"""
import numpy as np

from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.common import set_seed

set_seed(1)
np.random.seed(1)

class DeepIDWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(DeepIDWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, image, label):
        loss, _ = self.network(image, label)
        return loss


class TrainOneStepCell(nn.Cell):
    """Encapsulation class of StarGAN generator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph."""
    def __init__(self, net, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__()
        self.optimizer = optimizer
        self.net = net
        self.net.set_grad()
        self.net.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = DeepIDWithLossCell(net)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, image, label):
        weights = self.weights
        loss, acc = self.net(image, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(image, label, sens)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss, acc
