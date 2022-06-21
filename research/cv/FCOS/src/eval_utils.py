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
"""eval_utils"""
import numpy as np
import mindspore
import mindspore.numpy as mnp
from mindspore import ops, Tensor

def post_process(preds_topk, score_threshold, nms_iou_threshold):
    '''
    cls_scores_topk [batch_size,max_num]
    cls_classes_topk [batch_size,max_num]
    boxes_topk [batch_size,max_num,4]
    '''
    _cls_scores_post = []
    _cls_classes_post = []
    _boxes_post = []
    cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
    cls_scores_topk = mindspore.numpy.squeeze(cls_scores_topk, axis=0)
    cls_classes_topk = mindspore.numpy.squeeze(cls_classes_topk, axis=0)
    boxes_topk = mindspore.numpy.squeeze(boxes_topk, axis=0)
    for batch in range(cls_classes_topk.shape[0]):
        mask = cls_scores_topk[batch] >= score_threshold
        mul = ops.Mul()
        _cls_scores_b = mul(cls_scores_topk[batch], mask)
        _cls_scores_b = np.squeeze(_cls_scores_b)
        _cls_classes_b = mul(cls_classes_topk[batch], mask)
        _cls_classes_b = np.squeeze(_cls_classes_b)
        expand_dims = ops.ExpandDims()
        mask = expand_dims(mask, -1)
        op = ops.Concat(-1)
        mask = op((mask, mask, mask, mask))
        _boxes_b = mul(boxes_topk[batch], mask)
        nms_ind = batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, nms_iou_threshold)
        _cls_scores_post.append(_cls_scores_b[nms_ind])
        _cls_classes_post.append(_cls_classes_b[nms_ind])
        _boxes_post.append(_boxes_b[nms_ind])
    stack = ops.Stack(axis=0)
    scores, classes, boxes = stack(_cls_scores_post), stack(_cls_classes_post), stack(_boxes_post)
    return scores, classes, boxes

def batched_nms(boxes, scores, idxs, iou_threshold):
    if ops.Size()(boxes) == 0:
        return mnp.empty((0,))
    argmax = ops.ArgMaxWithValue()
    reshape = ops.Reshape()
    squeeze = ops.Squeeze()
    boxes2 = reshape(boxes, (-1, 1))
    _, max_coordinate = argmax(boxes2)
    max_coordinate = squeeze(max_coordinate)
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = box_nms(boxes_for_nms, scores, iou_threshold)
    return keep

def box_nms(boxes, scores, thr):
    '''
    boxes: [?,4]
    scores: [?]
    '''
    if boxes.shape[0] == 0:
        return ops.Zeros(0, mindspore.float32)
    boxes = boxes.asnumpy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sort = ops.Sort(0, descending=True)

    _, order = sort(scores)
    order = order.asnumpy()
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
        inds = np.where(diou <= 0.6)[0]
        order = order[inds + 1]
    return Tensor(keep, mindspore.int32)

def ClipBoxes(batch_imgs, batch_boxes):
    batch_boxes = ops.clip_by_value(batch_boxes, Tensor(0, mindspore.float32), Tensor(9999999, mindspore.float32))
    h, w = batch_imgs.shape[2:]
    batch_boxes[..., [0, 2]] = ops.clip_by_value(batch_boxes[..., [0, 2]], Tensor(0, mindspore.float32), w - 1)
    batch_boxes[..., [1, 3]] = ops.clip_by_value(batch_boxes[..., [1, 3]], Tensor(0, mindspore.float32), h - 1)
    return batch_boxes
