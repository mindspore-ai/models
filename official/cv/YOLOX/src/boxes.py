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
# =======================================================================================
"""
box iou related
"""
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr


@constexpr
def raise_bbox_error():
    raise IndexError("Index error, shape of input must be 4!")


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """
    calculate iou
    Args:
        bboxes_a:
        bboxes_b:
        xyxy:

    Returns:

    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise_bbox_error()

    if xyxy:
        tl = P.Maximum()(bboxes_a[:, None, :2], bboxes_b[:, :2])

        br = P.Minimum()(bboxes_a[:, None, 2:], bboxes_b[:, 2:])

        area_a = bboxes_a[:, 2:] - bboxes_a[:, :2]
        area_a = (area_a[:, 0:1] * area_a[:, 1:2]).squeeze(-1)

        area_b = bboxes_b[:, 2:] - bboxes_b[:, :2]
        area_b = (area_b[:, 0:1] * area_b[:, 1:2]).squeeze(-1)

    else:
        tl = P.Maximum()(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = P.Minimum()(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = (bboxes_a[:, 2:3] * bboxes_a[:, 3:4]).squeeze(-1)
        area_b = (bboxes_b[:, 2:3] * bboxes_b[:, 3:4]).squeeze(-1)
    en = (tl < br).astype(tl.dtype)
    en = (en[..., 0:1] * en[..., 1:2]).squeeze(-1)
    area_i = tl - br
    area_i = (area_i[:, :, 0:1] * area_i[:, :, 1:2]).squeeze(-1) * en
    return area_i / (area_a[:, None] + area_b - area_i)


def batch_bboxes_iou(batch_bboxes_a, batch_bboxes_b, xyxy=True):
    """
        calculate iou for one batch
    Args:
        batch_bboxes_a:
        batch_bboxes_b:
        xyxy:

    Returns:

    """
    if batch_bboxes_a.shape[-1] != 4 or batch_bboxes_b.shape[-1] != 4:
        raise_bbox_error()
    ious = []
    for i in range(len(batch_bboxes_a)):
        if xyxy:
            iou = bboxes_iou(batch_bboxes_a[i], batch_bboxes_b[i], True)
        else:
            iou = bboxes_iou(batch_bboxes_a[i], batch_bboxes_b[i], False)
        iou = P.ExpandDims()(iou, 0)
        ious.append(iou)
    ious = P.Concat(axis=0)(ious)
    return ious
