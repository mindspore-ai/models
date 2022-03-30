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
"""functions of criterion"""
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.mean_ops = P.ReduceMean(keep_dims=False)
        self.sum_ops = P.ReduceSum(keep_dims=False)
        self.log_softmax = P.LogSoftmax()

    def construct(self, logit, label):
        logit = P.Cast()(logit, mstype.float32)
        label = P.Cast()(label, mstype.float32)
        loss = self.sum_ops(-label * self.log_softmax(logit), -1)
        return self.mean_ops(loss)


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = ops.Cast()

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        label = P.Cast()(label, mstype.float32)
        logit = P.Cast()(logit, mstype.float32)
        loss2 = self.ce(logit, label)
        return loss2


def get_criterion(args):
    """Get loss function from args.label_smooth and args.mix_up"""
    assert args.label_smoothing >= 0. and args.label_smoothing <= 1.

    if args.mix_up > 0. or args.cutmix > 0.:
        print(25 * "=" + "Using MixBatch" + 25 * "=")
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.:
        print(25 * "=" + "Using label smoothing" + 25 * "=")
        criterion = CrossEntropySmooth(sparse=True, reduction="mean",
                                       smooth_factor=args.label_smoothing,
                                       num_classes=args.num_classes)
    else:
        print(25 * "=" + "Using Simple CE" + 25 * "=")
        criterion = CrossEntropySmooth(sparse=True, reduction="mean", num_classes=args.num_classes)

    return criterion


class NetWithLoss(nn.Cell):
    """
       NetWithLoss: Only support Network with Classfication
    """

    def __init__(self, model, criterion):
        super(NetWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def construct(self, data, label):
        predict = self.model(data)
        loss = self.criterion(predict, label)
        return loss
