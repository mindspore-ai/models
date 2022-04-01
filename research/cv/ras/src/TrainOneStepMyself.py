"""
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
"""

import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore as ms
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode


class TrainOneStep(nn.Cell):
    """
    Customize the training process
    """
    def __init__(self, network, optimizer, sens=1024):
        super(TrainOneStep, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.set_train()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.train_strategy = getattr(self.optimizer, 'train_strategy', None)
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reduce_flag = False

        self.parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reduce_flag = True
        if self.reduce_flag:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = ms.context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        else:
            print("-------------------- single card --------------------")
            self.grad_reducer = F.identity

    def construct(self, *inputs):
        """
        Args:
            *inputs: image,label,batchsize

        Returns:
            loss

        """
        image, label, batchsize = inputs
        loss = self.network(image, label)

        loss = loss / batchsize
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(image, label, sens)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
