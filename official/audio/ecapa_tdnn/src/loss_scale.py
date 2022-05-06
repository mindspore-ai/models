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
"""Loss scale cell for loss scale training."""

from mindspore import ops, nn
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore import RowTensor

_grad_scale = ops.composite.MultitypeFuncGraph("grad_scale")
reciprocal = ops.operations.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.functional.cast(reciprocal(scale), ops.functional.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * ops.functional.cast(reciprocal(scale), ops.functional.dtype(grad.values)),
                     grad.dense_shape)

_grad_overflow = ops.composite.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.operations.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


@_grad_overflow.register("RowTensor")
def _tensor_grad_overflow_row_tensor(grad):
    return grad_overflow(grad.values)

class ClipGradients(nn.Cell):
    """
    Clip gradients.
    Inputs:
        grads (tuple[Tensor]): Gradients.
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = ops.operations.Cast()
        self.dtype = ops.operations.DType()
    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        if clip_type not in(0, 1):
            return grads

        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = ops.composite.clip_by_value(grad, self.cast(ops.functional.tuple_to_array((-clip_value,)), dt),
                                                self.cast(ops.functional.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(ops.functional.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)
        return new_grads

class TrainOneStepWithLossScaleCellv2(TrainOneStepWithLossScaleCell):
    """
    Network training with loss scaling.
    """
    def __init__(self, network, optimizer, scale_sense):
        super(TrainOneStepWithLossScaleCellv2, self).__init__(
            network=network, optimizer=optimizer, scale_sense=scale_sense)
        self.clip_gradients = ClipGradients()
    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        output = self.network.output

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = ops.composite.ones_like(loss) * \
                             ops.functional.cast(scaling_sens, ops.functional.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.functional.partial(_grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, 0, 1.0)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = ops.functional.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens, output
