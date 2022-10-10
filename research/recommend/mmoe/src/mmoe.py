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
"""MMoE overall architecture"""
import mindspore
import mindspore.nn as nn
from mindspore import ops as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.communication.management import get_group_size
from mindspore.parallel._utils import _get_gradients_mean
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn import LossBase, BCELoss
from mindspore import Parameter, Tensor
from mindspore import context
from mindspore.context import ParallelMode

from src.model_utils.config import config
from src.mmoe_utils import expert, gate, shared_output, tower, output


class MMoE_Layer(nn.Cell):
    """MMoE network"""

    def __init__(self, input_size, num_experts, units):
        super(MMoE_Layer, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.units = units
        self.expert = expert(self.input_size, self.units, self.num_experts)
        self.gate1 = gate(self.input_size, self.num_experts)
        self.gate2 = gate(self.input_size, self.num_experts)
        self.shared_output1 = shared_output(
            self.input_size, self.num_experts, self.units)
        self.shared_output2 = shared_output(
            self.input_size, self.num_experts, self.units)
        self.tower_layer1 = tower(4, 8)
        self.tower_layer2 = tower(4, 8)
        self.output_layer1 = output(8, 2)
        self.output_layer2 = output(8, 2)
        self.concat = P.Concat(1)
        self.expand_dims = P.ExpandDims()
        self.print = P.Print()

    def construct(self, x):
        """construct of MMoE layer"""
        xx = self.expert(x)
        x1 = self.gate1(x)
        x2 = self.gate2(x)
        x1 = self.shared_output1(xx, x1)
        x2 = self.shared_output2(xx, x2)
        x1 = self.tower_layer1(x1)
        x1 = self.output_layer1(x1)
        x2 = self.tower_layer2(x2)
        x2 = self.output_layer2(x2)

        return x1, x2


def MMoE(num_features, num_experts, units):
    """MMoE call function"""
    net = MMoE_Layer(input_size=num_features,
                     num_experts=num_experts, units=units)

    return net


class LossForMultiLabel(LossBase):
    """loss for two labels"""

    def __init__(self):
        super(LossForMultiLabel, self).__init__()
        self.bceloss = BCELoss()

    def construct(self, base, target1, target2):
        base1 = base[0]
        base2 = base[1]
        x1 = self.bceloss(base1, target1)
        x2 = self.bceloss(base2, target2)
        # x1 = self.bceloss(base1, target1.astype(mindspore.float16))
        # x2 = self.bceloss(base2, target2.astype(mindspore.float16))
        return self.get_loss(x1) + self.get_loss(x2)


class NetWithLossClass(nn.Cell):
    """net with loss"""

    def __init__(self, model, loss_fn):
        super(NetWithLossClass, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def construct(self,
                  data,
                  income_labels,
                  married_labels):
        """construct"""
        out = self.model(data)
        loss = self.loss_fn(out, income_labels, married_labels)
        return loss

    @property
    def model_network(self):
        """get mmoe network"""
        return self.model


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5
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
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array(
            (-clip_value,)), dt), F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """scale grad """
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    """grad overflow """
    return grad_overflow(grad)


compute_norm = C.MultitypeFuncGraph("compute_norm")


@compute_norm.register("Tensor")
def _compute_norm(grad):
    norm = nn.Norm()
    norm = norm(F.cast(grad, mindspore.float32))
    ret = F.expand_dims(F.cast(norm, mindspore.float32), 0)
    return ret


grad_div = C.MultitypeFuncGraph("grad_div")


@grad_div.register("Tensor", "Tensor")
def _grad_div(val, grad):
    div = P.RealDiv()
    mul = P.Mul()
    scale = div(1.0, val)
    ret = mul(grad, scale)
    return ret


class TrainStepWrap(nn.Cell):
    """TrainStepWrap definition"""

    def __init__(self, network, optimizer, scale_update_cell, device_target):  # 16384.0
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.add_flags(has_effect=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer

        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = 1.0
        self.fill = P.Fill()
        self.dtype = P.DType()
        self.get_shape = P.Shape()
        self.cast = P.Cast()
        self.concat = P.Concat()
        self.less_equal = P.LessEqual()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.greater = P.Greater()
        self.select = P.Select()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.is_distributed = False
        self.norm = nn.Norm(keep_dims=True)
        self.base = Tensor(1, mindspore.float32)

        self.all_reduce = P.AllReduce()

        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = Parameter(
            Tensor(
                scale_update_cell.get_loss_scale(),
                dtype=mindspore.float32))

        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
                ParallelMode.DATA_PARALLEL,
                ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
            self.is_distributed = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            mean = _get_gradients_mean()
            self.grad_reducer = DistributedGradReducer(
                self.weights, mean, self.degree)
        self.device_target = device_target
        self.grad_scale_sense_type = mindspore.float16 \
            if config.device_target == 'Ascend' else mindspore.float32

    def construct(self, data, label1, label2):
        """construct"""
        weights = self.weights
        loss = self.network(data, label1, label2)

        scale_sense = self.loss_scale

        if self.device_target == 'Ascend':
            init = self.alloc_status()
            init = F.depend(init, loss)

            clear_status = self.clear_before_grad(init)
            scale_sense = F.depend(scale_sense, clear_status)
        else:
            init = False

        grads = self.grad(
            self.network,
            weights)(
                data,
                label1,
                label2,
                scale_sense.astype(self.grad_scale_sense_type))
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(
                grad_scale,
                self.degree *
                scale_sense),
            grads)
        grads = self.hyper_map(
            F.partial(
                clip_grad,
                GRADIENT_CLIP_TYPE,
                GRADIENT_CLIP_VALUE),
            grads)

        if self.device_target == 'Ascend':
            init = F.depend(init, grads)
            get_status = self.get_status(init)
            init = F.depend(init, get_status)
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = P.AddN()(flag_sum)
            flag_sum = P.Reshape()(flag_sum, (()))

        if self.is_distributed:
            flag_reduce = self.all_reduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        overflow = self.loss_scaling_manager(self.loss_scale, cond)

        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)

        ret = (loss, scale_sense.value())
        return F.depend(ret, succ)
