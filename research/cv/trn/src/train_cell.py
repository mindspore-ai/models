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
"""Implementation of a custom TrainOneStepCell with global gradients clipping"""

from mindspore import context
from mindspore import nn
from mindspore import ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_device_num


class CustomWithLossCell(nn.Cell):
    """Custom cell wrapper for attaching a loss function"""

    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, images, combinations, labels):
        """Build a feed forward graph"""
        network_output = self.network(images, combinations)
        return self.loss(network_output, labels)


class CustomTrainOneStepCell(nn.Cell):
    """Custom TrainOneStepCell with global gradients clipping"""

    def __init__(self, network, optimizer, max_grad_norm):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.reducer_flag = False
        self.grad_reducer = None
        self.max_grad_norm = max_grad_norm
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = _get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *inputs):
        """construct"""
        pred = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        pred = ops.depend(pred, self.optimizer(grads))
        return pred
