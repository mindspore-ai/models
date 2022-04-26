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
"""grad ops"""
import numpy as np


def area_box_grad(box):
    """box area grad"""
    return np.stack([
        box[:, 1] - box[:, 3],
        box[:, 0] - box[:, 2],
        box[:, 3] - box[:, 1],
        box[:, 2] - box[:, 0]
    ], axis=1)


def grad_xywh_to_cxcy(arr):
    """xywh to cxcy grad"""
    return np.stack([
        arr[:, 0] + arr[:, 2],
        arr[:, 1] + arr[:, 3],
        (-arr[:, 0] + arr[:, 2]) / 2,
        (-arr[:, 1] + arr[:, 3]) / 2
    ], axis=1)


def dres_dwh(dres_df, wh, is_maxmin,
             src_boxes_cxcy, target_boxes_cxcy):
    """dres/dwh"""
    d_dwh = np.stack([dres_df * wh[:, :, 1], dres_df * wh[:, :, 0]], axis=2)
    if is_maxmin:
        dwh_dmax = -d_dwh * (wh > 0.).astype(np.int32)
        dwh_dmin = d_dwh * (wh > 0.).astype(np.int32)
        max_grad_src = (src_boxes_cxcy[:, :2] > target_boxes_cxcy[:, :2]).astype(np.int32)
        min_grad_src = (src_boxes_cxcy[:, 2:] < target_boxes_cxcy[:, 2:]).astype(np.int32)
        src_grad = np.concatenate([
            (dwh_dmax * max_grad_src).sum(axis=1),
            (dwh_dmin * min_grad_src).sum(axis=1)
        ], axis=1)
    else:
        dwh_dmax = d_dwh * (wh > 0.).astype(np.int32)
        dwh_dmin = -d_dwh * (wh > 0.).astype(np.int32)
        max_grad_src = (src_boxes_cxcy[:, 2:] > target_boxes_cxcy[:, 2:]).astype(np.int32)
        min_grad_src = (src_boxes_cxcy[:, :2] < target_boxes_cxcy[:, :2]).astype(np.int32)
        src_grad = np.concatenate([
            (dwh_dmin * min_grad_src).sum(axis=1),
            (dwh_dmax * max_grad_src).sum(axis=1)
        ], axis=1)

    return src_grad


def grad_l1(src_boxes, tgt_boxes):
    """grad for l1"""
    neq_mask = (src_boxes != tgt_boxes).astype(np.float32)
    n_boxes = src_boxes.shape[0]
    grad_src = (-np.ones_like(src_boxes) + 2 * (src_boxes >= tgt_boxes)) * neq_mask / n_boxes
    return grad_src
