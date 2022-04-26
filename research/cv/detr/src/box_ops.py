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
"""box ops"""
import numpy as np

from src import grad_ops


def box_xyxy_to_cxcywh(x):
    """box xyxy to cxcywh"""
    x0, y0, x1, y1 = np.array_split(x.T, 4)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)[0]


def box_cxcywh_to_xyxy(x):
    """box cxcywh to xyxy"""
    x_c, y_c, w, h = np.array_split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1).squeeze(-2)


def box_xyxy_to_xywh(boxes):
    """box xyxy to xywh"""
    xmin, ymin, xmax, ymax = np.array_split(boxes.T, 4)
    return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=-1)[0]


def box_area(box):
    """box area"""
    return (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])


def box_iou(boxes1, boxes2):
    """box iou"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union, inter, wh


def generalized_box_iou(boxes1, boxes2, calc_grad=False):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union, inter, wh1 = box_iou(boxes1, boxes2)

    lt = np.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area

    if calc_grad:
        n_boxes = boxes1.shape[0]
        ddiagres_dres = -np.eye(n_boxes)
        dres_diou = ddiagres_dres
        dres_darea = ddiagres_dres * (-union / area ** 2)
        diou_dunion = dres_diou * (-inter / union ** 2)
        dres_dunion = diou_dunion + ddiagres_dres * (1 / area)
        dres_dinter = - dres_dunion + dres_diou / union

        src1 = grad_ops.dres_dwh(dres_dinter, wh1, True, boxes1, boxes2)

        dunion_darea1 = dres_dunion
        src2 = dunion_darea1.sum(axis=1, keepdims=True) * grad_ops.area_box_grad(boxes1)

        src3 = grad_ops.dres_dwh(dres_darea, wh, False, boxes1, boxes2)
        src_grad = grad_ops.grad_xywh_to_cxcy(src1 + src2 + src3) / n_boxes

        return giou, src_grad
    return giou
