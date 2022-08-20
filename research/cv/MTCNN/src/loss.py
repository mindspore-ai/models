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

from mindspore import nn, ops
import mindspore
import numpy

class ClassLoss(nn.Cell):
    def __init__(self, valid_class_num=2):
        super(ClassLoss, self).__init__()
        self.one_hot = nn.OneHot(depth=valid_class_num)
        self.keep_ratio = 0.7
        self.sort_descending = ops.Sort(descending=True)
        self.reduce_sum = ops.ReduceSum()
        self.stack = ops.Stack()
        self.gather = ops.Gather()

    def construct(self, gt_label, class_out):
        """
        gt_label: shape=(B)
        class_out: shape=(B, 2)
        """
        # Keep neg 0 and pos 1 data, ignore part -1, landmark -2
        valid_label = ops.select(gt_label >= 0, 1, ops.zeros_like(gt_label))
        num_valid = valid_label.sum()
        valid_class_out = class_out * valid_label.expand_dims(-1)
        keep_num = (num_valid * self.keep_ratio).astype(mindspore.int32)
        one_hot_label = self.one_hot(gt_label * valid_label)
        loss = ops.SoftmaxCrossEntropyWithLogits()(valid_class_out, one_hot_label)[0] * \
               valid_label.astype(valid_class_out.dtype)

        value, _ = self.sort_descending(loss)
        min_score = value[keep_num]
        mask = self.cast(loss > min_score, mindspore.float32)
        mask = ops.stop_gradient(mask)

        return self.reduce_sum(loss * mask) / keep_num


class BoxLoss(nn.Cell):
    def __init__(self):
        super(BoxLoss, self).__init__()
        self.loss_box = nn.MSELoss(reduction='none')
        self.abs = ops.Abs()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, gt_label, gt_offset, pred_offset):
        # Keep pos 1 and part -1
        valid_label = ops.select(self.abs(gt_label) == 1, 1, ops.zeros_like(gt_label))

        keep_num = valid_label.sum()
        loss = self.loss_box(pred_offset, gt_offset)
        loss = loss.sum(axis=1)
        loss = loss * valid_label
        # top k
        return self.reduce_sum(loss) / keep_num

class LandmarkLoss(nn.Cell):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.loss_landmark = nn.MSELoss(reduction='none')
        self.reduce_sum = ops.ReduceSum()

    def construct(self, gt_label, gt_landmark, pred_landmark):
        # Keep landmark -2
        valid_label = ops.select(gt_label == -2, 1, ops.zeros_like(gt_label))

        keep_num = valid_label.sum()
        loss = self.loss_landmark(pred_landmark, gt_landmark)
        loss = loss.sum(axis=1)
        loss = loss * valid_label

        return self.reduce_sum(loss) / keep_num

# Calculate accuracy while training
def accuracy(pred_label, gt_label):
    pred_label = pred_label.asnumpy()
    pred_label = pred_label.argmax(axis=1)
    gt_label = gt_label.asnumpy()
    zeros = numpy.zeros(gt_label.shape)
    cond = numpy.greater_equal(gt_label, zeros)
    picked = numpy.where(cond)
    valid_gt_label = gt_label[picked]
    valid_pred_label = pred_label[picked]

    acc = numpy.sum(valid_pred_label == valid_gt_label, dtype=numpy.float32) / valid_gt_label.shape[0]
    return acc
