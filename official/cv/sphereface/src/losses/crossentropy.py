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
loss function CrossEntropy
"""

import mindspore.common.initializer
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, Parameter


class AngleLoss(nn.Cell):
    def __init__(self, gamma=0, classnum=1009, smooth_factor=0.0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = Parameter(Tensor(0, mindspore.float32))
        self.LambdaMin = 5.0
        self.time = 0
        self.LambdaMax = Parameter(Tensor(1500, mindspore.float32))
        self.lamb = Parameter(Tensor(1500, mindspore.float32))
        self.on_value = Tensor(1.0 - smooth_factor, mindspore.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (classnum - 1), mindspore.float32)
    def construct(self, inputs, target):
        self.it += 1.0
        cos_theta = inputs[0]
        phi_theta = inputs[1]
        onehot = ops.OneHot()
        one_hot_label = onehot(target, ops.functional.shape(cos_theta)[1], self.on_value, self.off_value)
        if self.it < 3000:
            lamb = 1500 / (1 + 0.1 * self.it)
        else:
            lamb = 5
        output = cos_theta * 1.0
        cos_theta = one_hot_label * cos_theta * (1.0 / (1 + lamb))
        phi_theta = one_hot_label * phi_theta * (1.0 / (1 + lamb))
        output = output - cos_theta + phi_theta
        closs = mindspore.nn.SoftmaxCrossEntropyWithLogits()
        loss = closs(output, one_hot_label)
        cmean = ops.ReduceMean(False)
        loss = cmean(loss, 0)
        return loss
