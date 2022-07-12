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

"""deep supervision module"""

import mindspore
from mindspore import nn, Tensor


class MultipleOutputLoss2(nn.Cell):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = Tensor(weight_factors, mindspore.float32)
        self.loss = loss

    def construct(self, x, y_0, y_1, y_2):
        """construct deepsupervision"""
        weights = self.weight_factors
        l = weights[0] * self.loss(x[0], y_0)
        l += weights[1] * self.loss(x[1], y_1)
        l += weights[2] * self.loss(x[2], y_2)

        return l
