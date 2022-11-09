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

# This file was copied from third_party_mindspore [openharmony] [third_party_mindspore]
"""TrainOneStepCell"""
import mindspore.ops.functional as F
from mindspore import nn, ops

from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)

class TrainOneStepCell(nn.Cell):
    """Encapsulation class of network training."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        # self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)

        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, inputs):
        """construct"""
        loss = self.network(inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
