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
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.nn import rearrange_inputs
import numpy as np

class FlowNetEPE(nn.Metric):
    def __init__(self):
        super(FlowNetEPE, self).__init__()
        self.norm_op = nn.Norm(axis=1)
        self.mean = ops.ReduceMean()

    def clear(self):
        self._abs_error_sum = []
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('The MAE needs 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = self.mean(self.norm_op(ms.Tensor(y) - ms.Tensor(y_pred)))
        self._abs_error_sum.append(abs_error_sum.asnumpy().sum())
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('The total number of samples must not be 0.')
        return np.array(self._abs_error_sum).mean()
