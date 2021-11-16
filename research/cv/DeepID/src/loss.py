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
"""Loss Function."""
from __future__ import division

import mindspore
import mindspore.nn as nn
import mindspore.ops as P


class softmax_loss(nn.Cell):
    "Softmax Loss"
    def __init__(self):
        super(softmax_loss, self).__init__()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, predict, label):
        "Softmax Loss"
        loss = self.loss(predict, label)
        return loss


def cal_acc(predict, label):
    argmax = P.Argmax(axis=1)
    equal = P.Equal()
    p = argmax(predict)

    result = P.cast(equal(p, label), mindspore.float16)
    acc = (P.reduce_sum(result))/predict.shape[0]
    return acc


class DeepIDLoss(nn.Cell):
    """
    Deep ID Loss Cell
    """
    def __init__(self, net):
        super(DeepIDLoss, self).__init__()
        self.net = net
        self.loss = softmax_loss()

    def construct(self, image, label):
        "DeepID Loss"
        predict = self.net(image)
        loss = self.loss(predict, label)
        acc = cal_acc(predict, label)

        return (loss, acc)
