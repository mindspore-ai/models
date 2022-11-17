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
"""cell define"""
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore import ops
import mindspore

class WithLossCell(nn.Cell):
    """GenWithLossCell"""
    def __init__(self, net, criterion, auto_prefix=True):
        super(WithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.loss_fn = criterion

    def construct(self, patch, reward):
        """adnet construct"""
        fc6_out_, _ = self.net(patch, -1, False)
        # loss
        action = ops.Argmax(1, mindspore.dtype.int32)(fc6_out_)
        log_prob = ops.Log()(fc6_out_[:, action])
        loss = self.loss_fn(log_prob, reward)
        return loss

class TrainOneStepCell(nn.Cell):
    """define TrainOneStepCell"""
    def __init__(self, net, optimizer, sens=1.0, auto_prefix=True):

        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()
        self.net.add_flags(defer_inline=True)

        self.weights = optimizer.parameters
        self.optimizer = optimizer

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL,
                                  ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(
                self.weights, mean, degree)

    def construct(self, patch, reward):
        """construct"""
        loss = self.net(patch, reward)
        weights = self.weights
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, weights)(patch, reward, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
