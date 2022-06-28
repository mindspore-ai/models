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
"""define loss function for network"""
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor, ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter


class CrossEntropy(_Loss):
    """the redefined loss function with SoftmaxCrossEntropyWithLogits"""

    def __init__(self, smooth_factor=0, num_classes=5):
        super(CrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Parameter(Tensor(1.0 - smooth_factor, dtype=ms.float32))
        self.off_value = Parameter(Tensor(1.0 * smooth_factor / (num_classes - 1), dtype=ms.float32))
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        return loss
