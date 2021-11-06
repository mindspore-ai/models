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
"""Functions of optimizer"""

from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore import numpy as mnp
from mindspore import ops
from mindspore._checkparam import Validator
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Optimizer
from mindspore.ops import functional as F, composite as C

_momentum_opt = C.MultitypeFuncGraph("momentum_opt")


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment, ps_parameter, cache_enable):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    if ps_parameter and not cache_enable:
        op_shape = ops.Shape()
        _ps_pull = ops.Pull()
        _ps_push = ops.Push("ApplyMomentum", [])
        shapes = (op_shape(learning_rate), op_shape(gradient), op_shape(momentum))
        success = F.depend(True, _ps_pull(_ps_push((learning_rate, gradient, momentum), shapes), weight))
    else:
        success = F.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success


_agc_clip = C.MultitypeFuncGraph("agc_clip")


@_agc_clip.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool")
def _tensor_run_agc_ext(eps, min_grad, clipping, grad_norm, gradient, weight_norm, clip):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    if clip:
        param_norm = ops.Maximum()(weight_norm, eps)
        max_norm = param_norm * clipping
        trigger_mask = ops.Greater()(grad_norm, max_norm)
        clipped_grad = gradient * (max_norm / ops.Maximum()(grad_norm, min_grad))
        gradient = mnp.where(trigger_mask, clipped_grad, gradient)
    return gradient


_unitwise_norm = C.MultitypeFuncGraph("unitwise_norm")


@_unitwise_norm.register("Tensor")
def unitwise_norm_solve(x):
    """unitwise_norm_solve for weight"""
    if (len(ops.Squeeze()(x).shape)) <= 1:  # Scalars, vectors
        axis = 0
        keepdims = False
    elif len(x.shape) in [2, 3]:  # Linear layers
        # Original code: IO
        # Pytorch: OI
        axis = 1
        keepdims = True
    else:
        # Conv kernels
        # Original code: HWIO
        # Pytorch: OIHW
        axis = (1, 2, 3)
        keepdims = True
    return ops.Sqrt()(ops.ReduceSum(keepdims)(ops.Square()(x), axis))


class SGDAGC(Optimizer):
    """
    group_params = [{'params': conv_params,'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
    The final parameters order in which the optimizer will be followed is the value of 'order_params'.
    """

    def __init__(self, params, learning_rate=1e-3, momentum=0.9, eps=1e-3, clipping=None, weight_decay=2e-5,
                 use_nesterov=True):
        super(SGDAGC, self).__init__(learning_rate=learning_rate, parameters=params, weight_decay=weight_decay)

        Validator.check_bool(use_nesterov)

        self.use_nesterov = use_nesterov
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.eps = Parameter(Tensor(eps, mstype.float32), name="eps")

        assert clipping is not None, "For AGC_SGD, the clipping can't be None"
        self.clipping = Tensor(clipping, mstype.float32)
        self.min_grad = Tensor(1e-6, mstype.float32)

        self.group_clipping_tuple = []
        for param in self.group_params:
            name = param.name
            if not "attn_last" in name and "fc" in name and 'bias' not in name:
                print(f"{name} no clipping")
                self.group_clipping_tuple.append(False)
            else:
                self.group_clipping_tuple.append(True)
        self.group_clipping_tuple = tuple(self.group_clipping_tuple)
        assert len(self.group_clipping_tuple) == self.param_length

        self.moments = self.parameters.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = ops.ApplyMomentum(use_nesterov=self.use_nesterov)

    def construct(self, gradients):
        """agc_sgd construct"""
        # 1. get params and moments
        params = self.parameters
        moments = self.moments
        # 2. apply _unitwise_norm to gradients
        weight_norm = self.hyper_map(F.partial(_unitwise_norm), params)
        grad_norm = self.hyper_map(F.partial(_unitwise_norm), gradients)
        gradients = self.hyper_map(F.partial(_agc_clip, self.eps, self.min_grad, self.clipping), grad_norm,
                                   gradients, weight_norm, self.group_clipping_tuple)
        # 3. weight_decay
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        # 4. apply update, see mindspore.nn.Momentum, the ApplyMomentum has include the use_nesterov's function
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum), lr, gradients, params, moments,
                                     self.ps_parameters, self.cache_enable)
        else:
            success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments,
                                     self.ps_parameters, self.cache_enable)
        return success
