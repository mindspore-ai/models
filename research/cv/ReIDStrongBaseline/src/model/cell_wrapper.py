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
"""Cell_wrapper."""

from mindspore import ParameterTuple
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.communication import get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import Cell
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import composite as C
from mindspore.ops import functional as F


class NetworkWithCell(nn.Cell):
    """Build train network."""

    def __init__(self, network, criterion):
        super().__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        """ Forward """
        logits = self.network(input_data)
        loss = self.criterion(logits, label)
        return loss


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """ grad scale """
    return grad * F.cast(scale, F.dtype(grad))


class TrainOneStepCell(Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizers (Union[Cell, List[Cell]]): Optimizer for updating the weights.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.
        center_loss_weight (float): center loss weight (reverse center loss grad multiplier)

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If `sens` is not a number.
    """

    def __init__(self, network, optimizers, sens=1.0, center_loss_weight=1.):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer_net, self.optimizer_cri = optimizers
        self.weights_net = self.optimizer_net.parameters
        self.weights_cri = self.optimizer_cri.parameters
        self.weights = ParameterTuple(self.weights_net + self.weights_cri)

        self.num_weights_net = len(self.weights_net)
        self.cri_mult = Tensor(1. / center_loss_weight).astype('float32')
        self.hyper_map = C.HyperMap()

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = context.get_auto_parallel_context('parallel_mode')
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = context.get_auto_parallel_context('gradients_mean')
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        """ Forward """
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)

        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)

        grads_net, grads_cri = grads[:self.num_weights_net], grads[self.num_weights_net:]
        grads_cri = self.hyper_map(F.partial(grad_scale, self.cri_mult), grads_cri)

        loss = F.depend(loss, self.optimizer_net(grads_net))
        loss = F.depend(loss, self.optimizer_cri(grads_cri))

        return loss
