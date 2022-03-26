"""
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
"""

import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops


class LossFn(nn.Cell):
    """
    a loss function
    """
    def __init__(self):
        super(LossFn, self).__init__()
        self.loss_ = ops.BCEWithLogitsLoss(reduction='mean')

    def construct(self, image, label):
        """

        Args:
            image: image
            label: gt

        Returns:
            a float number , loss

        """

        weight = ops.Ones()((image.shape), ms.float32)
        loss = self.loss_(image, label, weight, weight)
        image = ops.Sigmoid()(image)
        sum_op = ops.ReduceSum(keep_dims=False)
        inter = sum_op(image * label, (2, 3))
        union = sum_op(image + label, (2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)

        result = (loss + iou).mean()
        return result

class BceIouLoss(nn.Cell):
    """
    a loss function
    """
    def __init__(self, batchsize):
        super(BceIouLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.batchsize = batchsize
        self.sigmoid = nn.Sigmoid()
        self.sum = ops.ReduceSum()

    def iou(self, predict, target):
        pred = self.sigmoid(predict)
        inter = self.sum(pred * target, (2, 3))
        union = self.sum(pred + target, (2, 3))
        iou = 1-(inter+1)/(union-inter+1)
        return iou

    def construct(self, predict, target):
        iou1 = self.iou(predict[0], target)
        bce1 = self.bce(predict[0], target)
        loss1 = self.mean(iou1 + bce1)
        iou2 = self.iou(predict[1], target)
        bce2 = self.bce(predict[1], target)
        loss2 = self.mean(iou2 + bce2)
        iou3 = self.iou(predict[2], target)
        bce3 = self.bce(predict[2], target)
        loss3 = self.mean(iou3 + bce3)
        iou4 = self.iou(predict[3], target)
        bce4 = self.bce(predict[3], target)
        loss4 = self.mean(iou4 + bce4)
        iou5 = self.iou(predict[4], target)
        bce5 = self.bce(predict[4], target)
        loss5 = self.mean(iou5 + bce5)
        loss_fuse = loss1 + loss2 + loss3 + loss4 + loss5
        loss = loss_fuse / self.batchsize
        return loss


class BuildTrainNetwork(nn.Cell):
    """
    Calculate 5 losses and summarize
    """
    def __init__(self, network, loss_fn):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, image, label):
        """

        Args:
            image:  image
            label:  gt

        Returns:
            a float number , loss_sum / 5
        """
        out1, out2, out3, out4, out5 = self.network(image)
        loss1 = self.loss_fn(out1, label)
        loss2 = self.loss_fn(out2, label)
        loss3 = self.loss_fn(out3, label)
        loss4 = self.loss_fn(out4, label)
        loss5 = self.loss_fn(out5, label)

        loss = loss1 + loss2 + loss3 + loss4 + loss5

        return loss
