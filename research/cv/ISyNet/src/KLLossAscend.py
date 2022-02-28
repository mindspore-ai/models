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
from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import LossBase
from mindspore import ops


class KLwithCELoss(LossBase):
    """KL Loss with CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000, dml=2):
        super().__init__()
        print("KLwithCELoss, dml={}", dml)
        self.dml = dml
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce1 = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.ce2 = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.log_softmax1 = nn.LogSoftmax()
        self.softmax1 = nn.Softmax()
        self.log_softmax2 = nn.LogSoftmax()
        self.softmax2 = nn.Softmax()
        self.plog1 = ops.Log()
        self.plog2 = ops.Log()


    def construct(self, logits, labels):
        "KL Loss construct"
        base, target = logits, labels
        if self.sparse:
            target = self.onehot(target, ops.shape(base[0])[1], self.on_value, self.off_value)

        loss = 0
        loss += self.ce1(base[0], target)
        loss += self.ce2(base[1], target)
        kl_loss = 0
        ls1 = self.log_softmax1(base[1])
        s1 = self.softmax1(base[0])
        ls2 = self.log_softmax2(base[0])
        s2 = self.softmax2(base[1])
        kl_1 = s1*(self.plog1(s1)-ls1)
        kl_2 = s2*(self.plog2(s2)-ls2)

        kl_loss1 = kl_1.sum(axis=1).mean()
        kl_loss2 = kl_2.sum(axis=1).mean()
        kl_loss = (kl_loss1 + kl_loss2)/2
        loss += kl_loss

        return loss
