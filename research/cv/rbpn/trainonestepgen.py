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

"""TrainOnestepGen network"""

from mindspore import nn
from mindspore import ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.
    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainOnestepGen(nn.TrainOneStepCell):
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

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(TrainOnestepGen, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.enable_clip_grad = enable_clip_grad

    def construct(self, target, x, neighbor, flow):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(target, x, neighbor, flow)
        sens_g = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(target, x, neighbor, flow, sens_g)
        if self.enable_clip_grad:
            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        return ops.depend(loss, self.optimizer(grads))
