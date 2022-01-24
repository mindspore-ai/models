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
"""target assigner"""
import logging

import numpy as np
import numpy.random as npr

logger = logging.getLogger(__name__)


def unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items
    (of size count)
    """
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def create_target_np(all_anchors,
                     gt_boxes,
                     similarity_fn,
                     box_encoding_fn,
                     prune_anchor_fn=None,
                     gt_classes=None,
                     matched_threshold=0.6,
                     unmatched_threshold=0.45,
                     positive_fraction=None,
                     rpn_batch_size=300,
                     norm_by_num_examples=False,
                     box_code_size=7):
    """Modified from FAIR detectron.
    Args:
        all_anchors: [num_of_anchors, box_ndim] float tensor.
        gt_boxes: [num_gt_boxes, box_ndim] float tensor.
        similarity_fn: a function, accept anchors and gt_boxes, return
            similarity matrix(such as IoU).
        box_encoding_fn: a function, accept gt_boxes and anchors, return
            box encodings(offsets).
        prune_anchor_fn: a function, accept anchors, return indices that
            indicate valid anchors.
        gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
            start with 1.
        matched_threshold: float, iou greater than matched_threshold will
            be treated as positives.
        unmatched_threshold: float, iou smaller than unmatched_threshold will
            be treated as negatives.
        positive_fraction: [0-1] float or None. if not None, we will try to
            keep ratio of pos/neg equal to positive_fraction when sample.
            if there is not enough positives, it fills the rest with negatives
        rpn_batch_size: int. sample size
        norm_by_num_examples: bool. norm box_weight by number of examples, but
            I recommend to do this outside.
        box_code_size: int. box coder size
    Returns:
        labels, bbox_targets, bbox_outside_weights
    """
    total_anchors = all_anchors.shape[0]
    if prune_anchor_fn is not None:
        inds_inside = prune_anchor_fn(all_anchors)
        anchors = all_anchors[inds_inside, :]
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[inds_inside]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[inds_inside]
    else:
        anchors = all_anchors
        inds_inside = None
    num_inside = len(inds_inside) if inds_inside is not None else total_anchors

    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
    labels = np.empty((num_inside,), dtype=np.int32)
    gt_ids = np.empty((num_inside,), dtype=np.int32)
    labels.fill(-1)
    gt_ids.fill(-1)
    if np.array(gt_boxes).shape[0] > 0 and anchors.shape[0] > 0:
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside), anchor_to_gt_argmax]
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax,
                                                np.arange(anchor_by_gt_overlap.shape[1])]
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1
        anchors_with_max_overlap = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[anchors_with_max_overlap] = gt_inds_force
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        bg_inds = np.arange(num_inside)
    fg_inds = np.where(labels > 0)[0]
    fg_max_overlap = None
    if np.array(gt_boxes).shape[0] > 0 and anchors.shape[0] > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds]
    gt_pos_ids = gt_ids[fg_inds]
    if positive_fraction is not None:
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            fg_inds = np.where(labels > 0)[0]

        num_bg = rpn_batch_size - np.sum(labels > 0)
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
    else:
        if np.array(gt_boxes).shape[0] == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
    bbox_targets = np.zeros((num_inside, box_code_size), dtype=all_anchors.dtype)
    if np.array(gt_boxes).shape[0] > 0 and anchors.shape[0] > 0:
        bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[anchor_to_gt_argmax[fg_inds], :],
                                                   anchors[fg_inds, :])

    bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)
    if norm_by_num_examples:
        num_examples = np.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0

    if inds_inside is not None:
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    ret = {
        "labels": labels,
        "bbox_targets": bbox_targets,
        "bbox_outside_weights": bbox_outside_weights,
        "assigned_anchors_overlap": fg_max_overlap,
        "positive_gt_id": gt_pos_ids,
    }
    if inds_inside is not None:
        ret["assigned_anchors_inds"] = inds_inside[fg_inds]
    else:
        ret["assigned_anchors_inds"] = fg_inds
    return ret


class TargetAssigner:
    """target assigner"""

    def __init__(self,
                 box_coder,
                 anchor_generators,
                 region_similarity_calculator=None,
                 positive_fraction=None,
                 sample_size=512):
        self._region_similarity_calculator = region_similarity_calculator
        self._box_coder = box_coder
        self._anchor_generators = anchor_generators
        self._positive_fraction = positive_fraction
        self._sample_size = sample_size

    @property
    def box_coder(self):
        """box coder"""
        return self._box_coder

    def assign(self,
               anchors,
               gt_boxes,
               anchors_mask=None,
               gt_classes=None,
               matched_thresholds=None,
               unmatched_thresholds=None):
        """assign"""

        def similarity_fn(anchors, gt_boxes):
            """similarity fn"""
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self._region_similarity_calculator.compare(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            """box encoding fn"""
            return self._box_coder.encode(boxes, anchors)

        if anchors_mask is not None:
            prune_anchor_fn = lambda _: np.where(anchors_mask)[0]
        else:
            prune_anchor_fn = None

        return create_target_np(anchors,
                                gt_boxes,
                                similarity_fn,
                                box_encoding_fn,
                                prune_anchor_fn=prune_anchor_fn,
                                gt_classes=gt_classes,
                                matched_threshold=matched_thresholds,
                                unmatched_threshold=unmatched_thresholds,
                                positive_fraction=self._positive_fraction,
                                rpn_batch_size=self._sample_size,
                                norm_by_num_examples=False,
                                box_code_size=self.box_coder.code_size)

    def generate_anchors(self, feature_map_size):
        """generate anchors"""
        anchors_list = []
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]
        match_list, unmatch_list = [], []
        for anchor_generator, match_thresh, unmatch_thresh in zip(self._anchor_generators,
                                                                  matched_thresholds,
                                                                  unmatched_thresholds):
            anchors = anchor_generator.generate(feature_map_size)
            anchors = anchors.reshape([*anchors.shape[:3], -1, 7])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
        anchors = np.concatenate(anchors_list, axis=-2)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds
        }

    @property
    def num_anchors_per_location(self):
        """num anchors per location"""
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num
