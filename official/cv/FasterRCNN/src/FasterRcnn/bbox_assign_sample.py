# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""FasterRcnn positive and negative sample screening for RPN."""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor


class BboxAssignSample(nn.Cell):
    """
    Bbox assigner and sampler definition.

    Args:
        config (dict): Config.
        batch_size (int): Batchsize.
        num_bboxes (int): The anchor nums.
        add_gt_as_proposals (bool): add gt bboxes as proposals flag.

    Returns:
        Tensor, output tensor.
        bbox_targets: bbox location, (batch_size, num_bboxes, 4)
        bbox_weights: bbox weights, (batch_size, num_bboxes, 1)
        labels: label for every bboxes, (batch_size, num_bboxes, 1)
        label_weights: label weight for every bboxes, (batch_size, num_bboxes, 1)

    Examples:
        BboxAssignSample(config, 2, 1024, True)
    """

    def __init__(self, config, batch_size, num_bboxes, add_gt_as_proposals):
        super(BboxAssignSample, self).__init__()
        cfg = config
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.batch_size = batch_size

        self.neg_iou_thr = Tensor(cfg.neg_iou_thr, self.ms_type)
        self.pos_iou_thr = Tensor(cfg.pos_iou_thr, self.ms_type)
        self.min_pos_iou = Tensor(cfg.min_pos_iou, self.ms_type)
        self.zero_thr = Tensor(0.0, self.ms_type)

        self.num_bboxes = num_bboxes
        self.num_gts = cfg.num_gts
        self.num_expected_pos = cfg.num_expected_pos
        self.num_expected_neg = cfg.num_expected_neg
        self.add_gt_as_proposals = add_gt_as_proposals

        if self.add_gt_as_proposals:
            self.label_inds = Tensor(np.arange(1, self.num_gts + 1))

        self.concat = ops.Concat(axis=0)
        self.max_gt = ops.ArgMaxWithValue(axis=0)
        self.max_anchor = ops.ArgMaxWithValue(axis=1)
        self.sum_inds = ops.ReduceSum()
        self.iou = ops.IOU()
        self.greaterequal = ops.GreaterEqual()
        self.greater = ops.Greater()
        self.select = ops.Select()
        self.gatherND = ops.GatherNd()
        self.squeeze = ops.Squeeze()
        self.cast = ops.Cast()
        self.logicaland = ops.LogicalAnd()
        self.less = ops.Less()
        self.random_choice_with_mask_pos = ops.RandomChoiceWithMask(self.num_expected_pos)
        self.random_choice_with_mask_neg = ops.RandomChoiceWithMask(self.num_expected_neg)
        self.reshape = ops.Reshape()
        self.equal = ops.Equal()
        self.bounding_box_encode = ops.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
        self.scatterNdUpdate = ops.ScatterNdUpdate()
        self.scatterNd = ops.ScatterNd()
        self.logicalnot = ops.LogicalNot()
        self.tile = ops.Tile()
        self.zeros_like = ops.ZerosLike()

        self.assigned_gt_inds = Tensor(np.full(num_bboxes, -1, dtype=np.int32))
        self.assigned_gt_zeros = Tensor(np.array(np.zeros(num_bboxes), dtype=np.int32))
        self.assigned_gt_ones = Tensor(np.array(np.ones(num_bboxes), dtype=np.int32))
        self.assigned_gt_ignores = Tensor(np.full(num_bboxes, -1, dtype=np.int32))
        self.assigned_pos_ones = Tensor(np.array(np.ones(self.num_expected_pos), dtype=np.int32))

        self.check_neg_mask = Tensor(np.array(np.ones(self.num_expected_neg - self.num_expected_pos), dtype=np.bool))
        self.range_pos_size = Tensor(np.arange(self.num_expected_pos).astype(self.dtype))
        self.check_gt_one = Tensor(np.full((self.num_gts, 4), -1, dtype=self.dtype))
        self.check_anchor_two = Tensor(np.full((self.num_bboxes, 4), -2, dtype=self.dtype))

    def construct(self, gt_bboxes_i, gt_labels_i, valid_mask, bboxes, gt_valids):
        gt_bboxes_i = self.select(self.cast(self.tile(self.reshape(self.cast(gt_valids, ms.int32),
                                                                   (self.num_gts, 1)), (1, 4)), ms.bool_), gt_bboxes_i,
                                  self.check_gt_one)
        bboxes = self.select(self.cast(self.tile(self.reshape(self.cast(valid_mask, ms.int32),
                                                              (self.num_bboxes, 1)), (1, 4)), ms.bool_), bboxes,
                             self.check_anchor_two)
        overlaps = self.iou(bboxes, gt_bboxes_i)
        max_overlaps_w_gt_index, max_overlaps_w_gt = self.max_gt(overlaps)
        _, max_overlaps_w_ac = self.max_anchor(overlaps)

        neg_sample_iou_mask = self.logicaland(self.greaterequal(max_overlaps_w_gt, self.zero_thr),
                                              self.less(max_overlaps_w_gt, self.neg_iou_thr))
        assigned_gt_inds2 = self.select(neg_sample_iou_mask, self.assigned_gt_zeros, self.assigned_gt_inds)

        pos_sample_iou_mask = self.greaterequal(max_overlaps_w_gt, self.pos_iou_thr)
        assigned_gt_inds3 = self.select(pos_sample_iou_mask,
                                        max_overlaps_w_gt_index + self.assigned_gt_ones, assigned_gt_inds2)
        assigned_gt_inds4 = assigned_gt_inds3
        for j in range(self.num_gts):
            max_overlaps_w_ac_j = max_overlaps_w_ac[j:j + 1:1]
            overlaps_w_gt_j = self.squeeze(overlaps[j:j + 1:1, ::])

            pos_mask_j = self.logicaland(self.greaterequal(max_overlaps_w_ac_j, self.min_pos_iou),
                                         self.equal(overlaps_w_gt_j, max_overlaps_w_ac_j))

            assigned_gt_inds4 = self.select(pos_mask_j, self.assigned_gt_ones + j, assigned_gt_inds4)

        assigned_gt_inds5 = self.select(valid_mask, assigned_gt_inds4, self.assigned_gt_ignores)

        pos_index, valid_pos_index = self.random_choice_with_mask_pos(self.greater(assigned_gt_inds5, 0))

        pos_check_valid = self.cast(self.greater(assigned_gt_inds5, 0), self.ms_type)
        pos_check_valid = self.sum_inds(pos_check_valid, -1)
        valid_pos_index = self.less(self.range_pos_size, pos_check_valid)
        pos_index = pos_index * self.reshape(self.cast(valid_pos_index, ms.int32), (self.num_expected_pos, 1))

        pos_assigned_gt_index = self.gatherND(assigned_gt_inds5, pos_index) - self.assigned_pos_ones
        pos_assigned_gt_index = pos_assigned_gt_index * self.cast(valid_pos_index, ms.int32)
        pos_assigned_gt_index = self.reshape(pos_assigned_gt_index, (self.num_expected_pos, 1))

        neg_index, valid_neg_index = self.random_choice_with_mask_neg(self.equal(assigned_gt_inds5, 0))

        num_pos = self.cast(self.logicalnot(valid_pos_index), self.ms_type)
        num_pos = self.sum_inds(num_pos, -1)
        unvalid_pos_index = self.less(self.range_pos_size, num_pos)
        valid_neg_index = self.logicaland(self.concat((self.check_neg_mask, unvalid_pos_index)), valid_neg_index)

        pos_bboxes_ = self.gatherND(bboxes, pos_index)
        pos_gt_bboxes_ = self.gatherND(gt_bboxes_i, pos_assigned_gt_index)
        pos_gt_labels = self.gatherND(gt_labels_i, pos_assigned_gt_index)

        pos_bbox_targets_ = self.bounding_box_encode(pos_bboxes_, pos_gt_bboxes_)

        valid_pos_index = self.cast(valid_pos_index, ms.int32)
        valid_neg_index = self.cast(valid_neg_index, ms.int32)
        total_index = self.concat((pos_index, neg_index))
        pos_index = self.cast(pos_index, ms.int64)
        bbox_targets_total = self.scatterNd(pos_index, pos_bbox_targets_, (self.num_bboxes, 4))
        bbox_weights_total = self.scatterNd(pos_index, valid_pos_index, (self.num_bboxes,))
        labels_total = self.scatterNd(pos_index, pos_gt_labels, (self.num_bboxes,))
        total_valid_index = self.concat((valid_pos_index, valid_neg_index))
        label_weights_total = self.scatterNd(total_index, total_valid_index, (self.num_bboxes,))

        return bbox_targets_total, self.cast(bbox_weights_total, ms.bool_), \
               labels_total, self.cast(label_weights_total, ms.bool_)
