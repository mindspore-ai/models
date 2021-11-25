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
"""Match each prior box with the ground truth box"""
import mindspore.ops as P
import mindspore.nn as nn
import mindspore
from src.config import yolact_plus_resnet50_config as cfg

class match(nn.Cell):
    """Match"""
    def __init__(self):
        super(match, self).__init__()
        self.cast = P.Cast()
        self.argmaxwithV0 = P.ArgMaxWithValue(0)
        self.argmaxwithV1 = P.ArgMaxWithValue(1)
        self.scalarcast = P.ScalarCast()
        self.squeeze = P.Squeeze()
        self.concat = P.Concat(1)
        self.log = P.Log()
        self.squeeze0 = P.Squeeze(0)
        self.squeeze2 = P.Squeeze(2)
        self.expand_dims = P.ExpandDims()

        shape = (1, 128, 57744)
        self.broadcast_to1 = P.BroadcastTo(shape)
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.prod = P.ReduceProd()
        self.zeroslike = P.ZerosLike()
        self.select = P.Select()
        shape2 = (1, 128, 57744, 2)

        self.broadcast_to2 = P.BroadcastTo(shape2)
        self.update = self.expand_dims(mindspore.Tensor(2, mindspore.float16), 0)
        self.range_op_1 = mindspore.nn.Range(0, 128, 1)
        self.range_op_2 = mindspore.nn.Range(0, 57744, 1)

        self.transpose = P.Transpose()
        self.prem = (1, 0)
        shape_crowd = (1, 57744, 10)

        self.broadcast_to1_crowd = P.BroadcastTo(shape_crowd)
        shape2_crowd = (1, 57744, 10, 2)

        self.broadcast_to2_crowd = P.BroadcastTo(shape2_crowd)
        self.reducesum = P.ReduceSum()
        self.oneslike = P.OnesLike()
        self.logicalAnd = P.LogicalAnd()
        self.iou = P.IOU(mode="iou")
        self.iou_f = P.IOU(mode="iof")
        self.crowd_iou_threshold = cfg['crowd_iou_threshold']

    def point_form(self, boxes):

        return self.concat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                            boxes[:, :2] + boxes[:, 2:] / 2))  # xmax, ymax

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
            is simply the intersection over union of two boxes.  Here we operate on
            ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        use_batch = True
        if box_a.ndim == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]
        inter = self.intersect(box_a, box_b)  # 1 128 4 ，1 19248，4  ---> 1 128 19248
        area_a = (box_a[:, :, 2:3:1] - box_a[:, :, 0:1:1]) * (box_a[:, :, 3:4:1] - box_a[:, :, 1:2:1])
        area_a = self.broadcast_to1(area_a)
        area_b = self.expand_dims(self.squeeze2((box_b[:, :, 2:3:1] - box_b[:, :, 0:1:1]) *
                                                (box_b[:, :, 3:4:1] - box_b[:, :, 1:2:1])), 1)
        area_b = self.broadcast_to1(area_b)
        union = area_a + area_b - inter
        out = inter / area_a if iscrowd else inter / union

        return out if use_batch else self.squeeze(out)

    def intersect(self, box_a, box_b):

        max_xy = self.min(self.broadcast_to2(self.expand_dims(box_a[:, :, 2:], 2)),
                          self.broadcast_to2(self.expand_dims(box_b[:, :, 2:], 1)))
        min_xy = self.max(self.broadcast_to2(self.expand_dims(box_a[:, :, :2], 2)),
                          self.broadcast_to2(self.expand_dims(box_b[:, :, :2], 1)))
        min_tensor = self.zeroslike(max_xy - min_xy)
        wants = self.select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)
        return self.prod(wants, 3)

    def jaccard_crowd(self, box_a, box_b, iscrowd: bool = False):
        """Jaccard"""
        use_batch = True
        if box_a.ndim == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]
        inter = self.intersect_crowd(box_a, box_b)  # 1 128 4 ，1 19248，4  ---> 1 128 19248
        area_a = (box_a[:, :, 2:3:1] - box_a[:, :, 0:1:1]) * (box_a[:, :, 3:4:1] - box_a[:, :, 1:2:1])
        area_a = self.broadcast_to1_crowd(area_a)
        area_b = self.expand_dims(self.squeeze2((box_b[:, :, 2:3:1] - box_b[:, :, 0:1:1]) *
                                                (box_b[:, :, 3:4:1] - box_b[:, :, 1:2:1])), 1)
        area_b = self.broadcast_to1_crowd(area_b)
        union = area_a + area_b - inter
        out = inter / area_a if iscrowd else inter / union

        return out if use_batch else self.squeeze(out)

    def intersect_crowd(self, box_a, box_b):
        """intersect"""
        max_xy = self.min(self.broadcast_to2_crowd(self.expand_dims(box_a[:, :, 2:], 2)),
                          self.broadcast_to2_crowd(self.expand_dims(box_b[:, :, 2:], 1)))
        min_xy = self.max(self.broadcast_to2_crowd(self.expand_dims(box_a[:, :, :2], 2)),
                          self.broadcast_to2_crowd(self.expand_dims(box_b[:, :, :2], 1)))
        min_tensor = self.zeroslike(max_xy - min_xy)
        wants = self.select(min_tensor > max_xy - min_xy, min_tensor, max_xy - min_xy)
        return self.prod(wants, 3)

    def encode(self, matched, priors):
        """
        Encode bboxes matched with each prior into the format
        produced by the network. See decode for more details on
        this format. Note that encode(decode(x, p), p) = x.

        Args:
            - matched: A tensor of bboxes in point form with shape [num_priors, 4]
            - priors:  The tensor of all priors with shape [num_priors, 4]
        Return: A tensor with encoded relative coordinates in the format
                outputted by the network (see decode). Size: [num_priors, 4]
        """
        # Encode the coordinates corresponding to the gtbox and the coordinates corresponding to the a priori box
        variances = [0.1, 0.2]
        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = self.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = self.concat((g_cxcy, g_wh))  # [num_priors,4]

        return loc

    def construct(self, pos_thresh, neg_thresh, truths, priors, labels,
                  crowd_boxes):
        """Forward"""
        # truths 128 4
        # priors 19248 4
        decoded_priors = self.point_form(priors)
        overlaps = self.jaccard(truths, decoded_priors)
        overlaps = self.cast(overlaps, mindspore.float32)
        best_truth_idx, best_truth_overlap = self.argmaxwithV0(overlaps)

        best_truth_idx = self.cast(best_truth_idx, mindspore.float16)
        best_truth_overlap = self.expand_dims(best_truth_overlap, 0)
        best_truth_overlap = self.cast(best_truth_overlap, mindspore.float16)
        best_truth_idx = self.expand_dims(best_truth_idx, 0)

        x_idx = P.Tile()(P.ExpandDims()(self.range_op_1(), 1), (1, 57744))
        z_idx = P.Tile()(self.range_op_2(), (128, 1))
        minus_one = P.OnesLike()(overlaps) * -1

        for _ in range(overlaps.shape[0]):

            best_prior_idx, best_prior_overlap = self.argmaxwithV1(overlaps)
            idx_j, out_j = self.argmaxwithV0(best_prior_overlap)
            i = best_prior_idx[idx_j]
            overlaps = self.select(x_idx == idx_j, minus_one, overlaps)
            overlaps = self.select(z_idx == i, minus_one, overlaps)
            idx_j = self.cast(idx_j, mindspore.float16)
            out_j = self.expand_dims(out_j, 0)
            best_truth_overlap[::, i] = self.select(out_j > 0, self.update, best_truth_overlap[::, i])
            idx_j = self.expand_dims(idx_j, 0)
            best_truth_idx[::, i] = self.select(out_j > 0, idx_j, best_truth_idx[::, i])

        best_truth_idx = self.squeeze(best_truth_idx)
        best_truth_overlap = self.squeeze(best_truth_overlap)
        best_truth_idx = self.cast(best_truth_idx, mindspore.int32)
        matches = truths[best_truth_idx]  # Shape: [num_priors,4]
        conf = labels[best_truth_idx] + 1  # Shape: [num_priors]  value is false
        conf[best_truth_overlap < pos_thresh] = -1
        conf[best_truth_overlap < neg_thresh] = 0  # label as background

        loc = self.encode(matches, priors)
        best_truth_idx = self.cast(best_truth_idx, mindspore.int32)

        return loc, conf, best_truth_idx
