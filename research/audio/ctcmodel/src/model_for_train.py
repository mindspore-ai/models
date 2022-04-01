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

"""Customized loss with cell class and Train one step with global clip"""

import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._auto_parallel_context import auto_parallel_context


class WithCtcLossCell(nn.Cell):
    """
       Wraps the network and loss
       Args:
           _backbone(Cell): The training network.
           loss_fn(Cell): Loss Function
    """

    def __init__(self, backbone, loss_fn):
        super(WithCtcLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, feature, masks, label, seq_len):
        logit = self._backbone(feature, masks)
        return self._loss_fn(logit, label, seq_len)


class TrainingWrapper(nn.Cell):
    '''
     Wraps the network with an optimizer
      Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.
        use_global_norm: Whether to use global grad clip norm
        clip_global_norm_value : The value of global clip norm
    '''

    def __init__(self, network, optimizer, sens=1.0, use_global_norm=True, clip_global_norm_value=5.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = float(sens)
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
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

    def construct(self, feature, masks, label, seq_len):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(feature, masks, label, seq_len)
        sens = P.Fill()(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, weights)(feature, masks, label, seq_len, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
        self.optimizer(grads)
        return loss
