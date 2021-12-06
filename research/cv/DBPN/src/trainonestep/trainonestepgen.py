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

"""TrainOnestepGen network"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context


class TrainOnestepGen(nn.Cell):
    """TrainOnestepGen
    Encapsulation class of DBPN network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        network(Cell): Generator with loss Cell. Note that loss function should have been added
        optimizer(Cell):Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    Outputs:
        Tensor
    """

    def __init__(self, loss, optimizer, sens=1.0):
        super(TrainOnestepGen, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.loss = loss
        self.loss.set_grad()
        self.loss.set_train()
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

    def construct(self, HR_img, LR_img):
        """Defines the computation performed."""
        weights = self.weights
        content_loss = self.loss(HR_img, LR_img)
        sens = ops.Fill()(ops.DType()(content_loss), ops.Shape()(content_loss), self.sens)
        grads = self.grad(self.loss, weights)(HR_img, LR_img, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return ops.depend(content_loss, self.optimizer(grads))
