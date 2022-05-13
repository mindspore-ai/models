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

import math
import mindspore as ms
import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, MatMul
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

matMul_tb = MatMul(transpose_x2=True)


def linear(_input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    output = matMul_tb(_input, weight)
    if bias is not None:
        output += bias
    return output


class MaskerFill(Cell):
    def __init__(self):
        super(MaskerFill, self).__init__()
        self.select = P.Select()
        self.fill = P.Fill()
        self.cast = P.Cast()

    def construct(self, inputs, mask, value):
        mask = self.cast(mask, mstype.bool_)
        masked_value = self.fill(ms.float32, inputs.shape, value)
        output = self.select(mask, masked_value, inputs)
        return output


class CosineAnnealingLR(LearningRateSchedule):
    def __init__(self, total_step, lr, min_lr=0):
        super(CosineAnnealingLR, self).__init__()

        self.min_lr = Parameter(Tensor(min_lr, ms.float32))
        self.lr = Parameter(Tensor(lr, ms.float32))
        self.max_lr = Parameter(Tensor(lr, ms.float32))
        self.T_max = Parameter(Tensor(total_step, ms.float32))

        self.cos = P.Cos()
        self.pi = Parameter(Tensor(math.pi, ms.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        global_step = self.cast(global_step, ms.float32)
        if global_step <= 0:
            self.lr = self.max_lr
        elif (global_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            self.lr += (self.max_lr - self.min_lr) * (1 - self.cos(self.pi / self.T_max)) / 2
        else:
            self.lr = (1 + self.cos(self.pi * global_step / self.T_max)) /\
                      (1 + self.cos(self.pi * (global_step - 1) / self.T_max)) * (self.lr - self.min_lr) + self.min_lr

        return self.lr
