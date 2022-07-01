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
import src.utils as utils
import mindspore.ops as ops
from mindspore.nn import Metric
from mindspore.nn import LossBase
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.nn import rearrange_inputs
from mindspore.ops import operations as P


class MyMAE(Metric):

    def __init__(self, horizon, num_nodes, output_dim, batch_size):
        super(MyMAE, self).__init__()
        self.clear()
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.abs = ops.Abs()

    def clear(self):
        self._mae_error = 0  # Metric好像没有getloss这个方法

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        base = inputs[0] #y_pred
        target = inputs[1] #y_true

        base = Tensor(base, mstype.float32)
        mask = Tensor(target, mstype.float32)
        mask /= mask.mean()
        loss = self.abs(target - base)
        loss = loss * mask
        loss[0] = 0
        self._mae_error = loss.mean()
        self._mae_error /= 54

    def eval(self):
        return self._mae_error


class MaskedMAELoss(LossBase):
    def __init__(self, horizon, num_nodes, output_dim, batch_size, reduction="mean"):
        super(MaskedMAELoss, self).__init__(reduction)
        self.abs = ops.Abs()
        self.cast = P.Cast()
        self._data = utils.load_dataset()
        self.transpose = ops.Transpose()
        self.std = self._data['scaler'].get_std()
        self.mean = self._data['scaler'].get_mean()
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.batch_size = batch_size

    def construct(self, base, target):

        target = self.cast(target, mstype.float32)
        target = self.transpose(target, (1, 0, 2, 3))
        target = target[..., :self.output_dim].view(self.horizon, self.batch_size, self.num_nodes * self.output_dim)
        target = (target * self.std) + self.mean

        base = (base * self.std) + self.mean

        mask = target.copy() # Tensor(index, mstype.float32)
        mask /= mask.mean()
        loss = self.abs(base - target)
        loss = loss * mask
        loss[0] = 0
        return self.get_loss(loss)
