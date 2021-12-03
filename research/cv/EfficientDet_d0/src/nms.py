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
"""nms used for eval """
import numpy as np

def batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.shape[0] == 0:
        return np.empty((0,)).astype(np.int32)
    max_coordinate = boxes.max()       # returns max val in tensor
    offsets = idxs.astype(boxes.dtype) * (max_coordinate + np.array(1).astype(boxes.dtype))
    offsets = np.expand_dims(offsets, 1)
    boxes_for_nms = boxes + offsets
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

def _diou_nms(all_boxes, all_scores, thresh=0.5):
    """convert xywh -> xmin ymin xmax ymax"""
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    scores = all_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        center_x1 = (x1[i] + x2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        out_max_x = np.maximum(x2[i], x2[order[1:]])
        out_max_y = np.maximum(y2[i], y2[order[1:]])
        out_min_x = np.minimum(x1[i], x1[order[1:]])
        out_min_y = np.minimum(y1[i], y1[order[1:]])
        outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
        diou = ovr - inter_diag / outer_diag
        diou = np.clip(diou, -1, 1)
        inds = np.where(diou <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def nms(all_boxes, all_scores, thres):
    """Apply NMS to bboxes."""

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return np.array(keep)
