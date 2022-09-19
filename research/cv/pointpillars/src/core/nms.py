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
"""NMS"""
import numpy as np
from mindspore import ops
from mindspore import Tensor


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    all_boxes = all_boxes.asnumpy()
    all_scores = all_scores.asnumpy()
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

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


def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    """NMS"""
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = ops.TopK()(scores, pre_max_size)
        bboxes = bboxes[indices]

    keep = apply_nms(bboxes, scores, iou_threshold, post_max_size)
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        keep = Tensor(keep)
        return indices[keep]
    return Tensor(keep)


def topk_(matrix, K, axis=0):
    K = K - 1
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort

def nms_jit(dets, thresh, eps=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep

def nms_np(bboxes,
           scores,
           pre_max_size=None,
           post_max_size=None,
           iou_threshold=0.5):
    """NMS"""
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores = np.expand_dims(scores, 1)
        scores, indices = topk_(scores, pre_max_size)
        bboxes = bboxes[indices]
    bboxes = bboxes.reshape(bboxes.shape[0], bboxes.shape[-1])
    dets = np.concatenate([bboxes, scores], axis=1)   # for eval
    dets_np = dets
    if dets_np.size == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_jit(dets_np, iou_threshold), dtype=np.int64)  # cpu 加速（关）
        keep = ret[:post_max_size]

    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        return indices[keep].reshape(indices[keep].shape[0])
    return keep.reshape(keep.shape[0])
