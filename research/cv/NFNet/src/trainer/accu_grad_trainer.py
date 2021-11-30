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
"""Accumulation Gradients"""
import mindspore.nn as nn
from mindspore import Parameter, Tensor, RowTensor
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

_sum_op = C.MultitypeFuncGraph("grad_sum_op")
assignadd = P.AssignAdd()


@_sum_op.register("Tensor", "Tensor")
def _accu_grad(grad_sum, grad):
    """Apply grad sum to cumulative gradient."""

    return assignadd(grad_sum, grad)


_clear_op = C.MultitypeFuncGraph("grad_clear_op")
assign = P.Assign()


@_clear_op.register("Tensor", "Tensor")
def _clear_grad(grad_sum, zero):
    """Apply grad clear to cumulative gradient."""

    return assign(grad_sum, zero)


_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


class TrainOneStepWithLossScaleCellAccumulation(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer,
                 scale_sense=1.0, accumulation_step=8):
        super(TrainOneStepWithLossScaleCellAccumulation, self).__init__(network, optimizer, scale_sense)

        self.assignadd = P.AssignAdd()
        self.accumulation_step = int(accumulation_step)
        assert self.accumulation_step > 1
        self._grad_sum = optimizer.parameters.clone(prefix="grad_sum", init='zeros')
        self._zeros = optimizer.parameters.clone(prefix="zero", init='zeros')
        self.cur_step_num = Parameter(Tensor(0, mstype.int64), requires_grad=False)
        self.print = P.Print()

    def construct(self, *inputs):
        """TrainOneStepWithLossScaleCellAccumulation Construct Function"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        loss = loss / self.accumulation_step
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            self.hyper_map(F.partial(_sum_op), self._grad_sum, grads)
            self.assignadd(self.cur_step_num, 1)
            if self.cur_step_num % self.accumulation_step == 0:
                loss = F.depend(loss, self.optimizer(self._grad_sum))
                self.hyper_map(F.partial(_clear_op), self._grad_sum, self._zeros)
        else:
            self.print(self.cur_step_num, "=============Over Flow, skipping=============")
        return loss
