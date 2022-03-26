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

"""Train forward and backward define"""

from mindspore import ops
from mindspore.nn import Cell, TrainOneStepCell

_sum_op = ops.MultitypeFuncGraph("grad_sum_op")
_clear_op = ops.MultitypeFuncGraph("clear_op")


@_sum_op.register("Tensor", "Tensor")
def _cumulative_grad(grad_sum, grad):
    """Apply grad sum to cumulative gradient."""
    add = ops.AssignAdd()
    return add(grad_sum, grad)


@_clear_op.register("Tensor", "Tensor")
def _clear_grad_sum(grad_sum, zero):
    """Apply zero to clear grad_sum."""
    success = True
    success = ops.depend(success, ops.assign(grad_sum, zero))
    return success


class TrainForwardBackward(TrainOneStepCell):
    """
    cell for step train
    """
    def __init__(self, network, optimizer, grad_sum, sens=1.0):
        super(TrainForwardBackward, self).__init__(network, optimizer, sens)
        self.grad_sum = grad_sum
        self.sens = sens
        self.hyper_map = ops.HyperMap()

    def construct(self, *inputs):
        """
        forward one step, accumulate grad
        """
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        return ops.depend(loss, self.hyper_map(ops.partial(_sum_op), self.grad_sum, grads))


class TrainOptimize(Cell):
    """
    optimize cell
    """

    def __init__(self, optimizer, grad_sum):
        super(TrainOptimize, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.grad_sum = grad_sum

    def construct(self):
        """
        optimize
        :return:
        """
        return self.optimizer(self.grad_sum)


class TrainClear(Cell):
    """
    clear cell
    """

    def __init__(self, grad_sum, zeros):
        super(TrainClear, self).__init__(auto_prefix=False)
        self.grad_sum = grad_sum
        self.zeros = zeros
        self.hyper_map = ops.HyperMap()

    def construct(self):
        """
        clear grad
        """
        success = self.hyper_map(ops.partial(_clear_op), self.grad_sum, self.zeros)
        return success
