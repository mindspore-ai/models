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
import mindspore
from mindspore import nn, ops, Tensor

grad_scale = ops.MultitypeFuncGraph("grad_scale")

@grad_scale.register("Tensor", "Tensor")
def gradient_scale(scale, grad):
    return grad * ops.cast(scale, ops.dtype(grad))

class CustomTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(CustomTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = ops.HyperMap()
        self.reciprocal_sense = Tensor(1 / sens, mindspore.float32)

    def scale_grad(self, gradients):
        gradients = self.hyper_map(ops.partial(grad_scale, self.reciprocal_sense), gradients)
        return gradients

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        # calculate gradients, the sens will equal to the loss_scale
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        # gradients / loss_scale
        grads = self.scale_grad(grads)
        # reduce gradients in distributed scenarios
        grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
