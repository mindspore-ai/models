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
'''LOSS'''
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops

#loss function
class BinaryCrossEntropyLoss(nn.Cell):
    def __init__(self, model):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.model = model
        self.binary_cross_entropy = ops.BinaryCrossEntropy()
        self.sum = P.ReduceSum(False)
        self.cast = P.Cast()
        self.size = P.Size()
        self.zeros = ops.Zeros()

    def construct(self, x, get_label, mask_wh):
        mask = (get_label != 0)
        mask = self.cast(mask, mstype.float32)
        num_positive = self.sum(mask)
        num_ne_mask = self.size(mask_wh) - self.sum(mask_wh)
        num_negative = self.size(mask) - num_positive - num_ne_mask
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)

        loss = 0
        so1, so2, so3, so4, so5, fuse = self.model(x)
        so1 = so1 * mask_wh
        so2 = so2 * mask_wh
        so3 = so3 * mask_wh
        so4 = so4 * mask_wh
        so5 = so5 * mask_wh
        fuse = fuse * mask_wh
        get_label = get_label * mask_wh

        loss += self.binary_cross_entropy(so1, get_label, mask)
        loss += self.binary_cross_entropy(so2, get_label, mask)
        loss += self.binary_cross_entropy(so3, get_label, mask)
        loss += self.binary_cross_entropy(so4, get_label, mask)
        loss += self.binary_cross_entropy(so5, get_label, mask)
        loss += self.binary_cross_entropy(fuse, get_label, mask)
        return self.sum(loss)
