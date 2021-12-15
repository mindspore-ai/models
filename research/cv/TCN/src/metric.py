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
"""
######################## definition of metric ########################
"""
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.metrics.metric import Metric, rearrange_inputs


class MyLoss(Metric):
    """definition of metric"""

    def __init__(self):
        super(MyLoss, self).__init__()
        self.myloss = nn.MSELoss()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._abs_error_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = self.myloss(Tensor(y_pred), Tensor(y))
        self._abs_error_sum += abs_error_sum.sum()
        self._samples_num += 1

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num
