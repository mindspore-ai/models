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
"""bbox ops"""
import box_list_np as box_list
import numpy as np


def nms_op(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        IOU = inter / (areas[i] + areas[order[1:]] - inter)

        left_index = (np.where(IOU <= thresh))[0]

        order = order[left_index + 1]

    return np.array(keep)

def gather(boxlist, indices):
    """Gather boxes from BoxList according to indices and return new BoxList.

    Args:
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64

    Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
    specified by indices
    Raises:
    ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int32
    """
    subboxlist = box_list.BoxList(boxlist.get()[indices])
    fields = boxlist.get_extra_fields()
    for field in fields:
        subfieldlist = boxlist.get_field(field)[indices]
        subboxlist.add_field(field, subfieldlist)
    return subboxlist

def non_max_suppression(boxlist, thresh, max_output_size):
    """Non maximum suppression.

    This op greedily selects a subset of detection bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected boxes.  Note that this only works for a single class ---
    to apply NMS to multi-class predictions, use MultiClassNonMaxSuppression.

    Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
        representing detection scores.
    thresh: scalar threshold
    max_output_size: maximum number of retained boxes

    Returns:
    a BoxList holding M boxes where M <= max_output_size
    Raises:
    ValueError: if thresh is not in [0, 1]
    """
    if not 0 <= thresh <= 1.0:
        raise ValueError('thresh must be between 0 and 1')
    if not isinstance(boxlist, box_list.BoxList):
        raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
        raise ValueError('input boxlist must have \'scores\' field')

    bbox = boxlist.get()
    scores = boxlist.get_field('scores')
    scores = scores.reshape(-1, 1)
    bbox_new = np.concatenate((bbox, scores), 1)

    selected_indices = nms_op(bbox_new, thresh)

    scores = scores.reshape(-1)[selected_indices]
    Z = zip(scores, selected_indices)
    Z = sorted(Z, reverse=True)
    _, indices_new = zip(*Z)
    max_size = np.arange(max_output_size)
    indices_new_1 = np.array(indices_new)[max_size]

    return gather(boxlist, indices_new_1)
