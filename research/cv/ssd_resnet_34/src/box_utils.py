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

"""Bound box utils"""

import itertools as it
import math

import numpy as np

from .anchor_generator import GridAnchorGenerator
from .config import config


class GenerateDefaultBoxes:
    """
    Generate Default boxes for SSD, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_tlbr` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """
    def __init__(self):
        fk = config.img_shape[0] / np.array(config.steps)
        scale_rate = (config.max_scale - config.min_scale) / (len(config.num_default) - 1)
        scales = [config.min_scale + scale_rate * i for i in range(len(config.num_default))] + [1.0]
        self.default_boxes = []
        for index, feature_size in enumerate(config.feature_size):
            sk1 = scales[index]
            sk2 = scales[index + 1]
            sk3 = math.sqrt(sk1 * sk2)
            if index == 0 and not config.aspect_ratios[index]:
                width, height = sk1 * math.sqrt(2), sk1 / math.sqrt(2)
                all_sizes = [(0.1, 0.1), (width, height), (height, width)]
            else:
                all_sizes = [(sk1, sk1)]
                for aspect_ratio in config.aspect_ratios[index]:
                    width, height = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                    all_sizes.append((width, height))
                    all_sizes.append((height, width))
                all_sizes.append((sk3, sk3))

            assert len(all_sizes) == config.num_default[index]

            for i, j in it.product(range(feature_size), repeat=2):
                for width, height in all_sizes:
                    center_x, center_y = (j + 0.5) / fk[index], (i + 0.5) / fk[index]
                    self.default_boxes.append([center_y, center_x, height, width])

        def to_tlbr(cy, cx, h, w):
            """Convert boxes from [y, x, h, w] to [y1, x1, y2, x2] format."""
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_tlbr = np.array(tuple(to_tlbr(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')


if 'use_anchor_generator' in config and config.use_anchor_generator:
    generator = GridAnchorGenerator(config.img_shape, 4, 2, [1.0, 2.0, 0.5])
    default_boxes, default_boxes_tlbr = generator.generate_multi_levels(config.steps)
else:
    default_boxes_tlbr = GenerateDefaultBoxes().default_boxes_tlbr
    default_boxes = GenerateDefaultBoxes().default_boxes
y1, x1, y2, x2 = np.split(default_boxes_tlbr[:, :4], 4, axis=-1)
vol_anchors = (x2 - x1) * (y2 - y1)
matching_threshold = config.match_threshold


def ssd_bboxes_encode(boxes):
    """
    Labels anchors with ground truth inputs.

    Args:
        boxes: ground truth with shape [N, 5], for each row, it stores [y, x, h, w, cls].

    Returns:
        gt_loc: location ground truth with shape [num_anchors, 4].
        gt_label: class ground truth with shape [num_anchors, 1].
        num_matched_boxes: number of positives in an image.
    """

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bound_box and volume.
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros(config.num_ssd_boxes, dtype=np.float32)
    t_boxes = np.zeros((config.num_ssd_boxes, 4), dtype=np.float32)
    t_label = np.zeros(config.num_ssd_boxes, dtype=np.int64)
    for bound_box in boxes:
        label = int(bound_box[4])
        scores = jaccard_with_anchors(bound_box)
        idx = np.argmax(scores)
        scores[idx] = 2.0
        mask = (scores > matching_threshold)
        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores * mask)
        t_label = mask * label + (1 - mask) * t_label
        for i in range(4):
            t_boxes[:, i] = mask * bound_box[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)

    # Transform to tlbr.
    bboxes = np.zeros((config.num_ssd_boxes, 4), dtype=np.float32)
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

    # Encode features.
    bboxes_t = bboxes[index]
    default_boxes_t = default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / (default_boxes_t[:, 2:] * config.prior_scaling[0])
    tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
    bboxes_t[:, 2:4] = np.log(tmp) / config.prior_scaling[1]
    bboxes[index] = bboxes_t

    num_match = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    return bboxes, t_label.astype(np.int32), num_match


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union
