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
"""riemannian adam"""
import numpy as np
from mindspore import Parameter
import mindspore.common.dtype as mstype
from mindspore.common import Tensor
from mindspore.nn.optim.optimizer import opt_init_args_register, Optimizer
from mindspore.ops import Sqrt, Add, Assign, Pow, Mul, ReduceSum


class RiemannianAdam(Optimizer):
    """RiemannianAdam optimizer"""

    @opt_init_args_register
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, \
                 weight_decay=0.0):
        """init fun"""
        super(RiemannianAdam, self).__init__(learning_rate=learning_rate, parameters=params, weight_decay=weight_decay)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.sum = ReduceSum(keep_dims=True)
        self.sumFalse = ReduceSum(keep_dims=False)
        self.sqrt = Sqrt()
        self.add = Add()
        self.exp_avg = self.parameters.clone(prefix='exp_avg', init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix='exp_avg_sq', init='zeros')
        self.step = Parameter(Tensor(0, mstype.int32), name='step')
        self.assign = Assign()
        self.pow = Pow()
        self.mul = Mul()

    def construct(self, gradients):
        """class construction"""
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps
        learning_rate = self.get_lr()
        params = self.parameters
        success = None
        step = self.step
        for exp_avg, exp_avg_sq, param, grad in zip(self.exp_avg, self.exp_avg_sq, params, gradients):
            point = param
            if grad is None:
                continue
            exp_avg_update = self.add(self.mul(exp_avg, beta1), (1 - beta1) * grad)
            exp_avg_sq_update = self.add(self.mul(exp_avg_sq, beta2),
                                         (1 - beta2) * (self.sum(grad * grad, -1))
                                         )
            denom = self.add(self.sqrt(exp_avg_sq_update), eps)
            step += 1
            bias_cor1 = 1 - self.pow(beta1, step)
            bias_cor2 = 1 - self.pow(beta2, step)
            step_size = learning_rate * bias_cor2 ** 0.5 / bias_cor1
            direction = exp_avg_update / denom
            new_point = point - step_size * direction
            step += 1
            self.assign(exp_avg, exp_avg_update)
            self.assign(exp_avg_sq, exp_avg_sq_update)
            success = self.assign(param, new_point)
        self.assign(self.step, step)
        return success
