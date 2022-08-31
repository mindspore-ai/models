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

import mindspore
from mindspore import ops
import mindspore.nn as nn

class WithLossCellDP(nn.Cell):
    def __init__(self, CNet, options, auto_prefix=False):
        super(WithLossCellDP, self).__init__(auto_prefix=auto_prefix)
        self.CNet = CNet
        self.criterion_uv = nn.L1Loss()
        self.criterion_mask = nn.BCELoss(reduction='mean')
        self.options = options
        self.lam_dp_uv = options.lam_dp_uv
        self.lam_dp_mask = options.lam_dp_mask
        self.adaptive_weight = options.adaptive_weight
        self.expand_dims = ops.ExpandDims()

    def error_adaptive_weight(self, fit_joint_error):
        weight = (1 - 10 * fit_joint_error)
        weight[weight <= 0] = 0
        return weight

    def dp_loss(self, pred_dp, gt_dp, has_dp, weight=None):

        pred_dp_shape = pred_dp * has_dp.astype(pred_dp.dtype).mean()
        gt_dp_shape = gt_dp * has_dp.astype(pred_dp.dtype).mean()

        if gt_dp_shape.shape[0] > 0:
            gt_dp_shape_temp = self.expand_dims(gt_dp_shape[:, 0], 1)
            gt_mask_shape = gt_dp_shape_temp > 0
            gt_mask_shape = gt_mask_shape.astype(pred_dp.dtype)

            gt_uv_shape = gt_dp_shape[:, 1:]

            pred_mask_shape = self.expand_dims(pred_dp_shape[:, 0], 1)
            pred_uv_shape = pred_dp_shape[:, 1:]

            interpolate_bilinear = ops.ResizeBilinear((gt_dp.shape[2], gt_dp.shape[3]))
            interpolate_nearest = ops.ResizeNearestNeighbor((gt_dp.shape[2], gt_dp.shape[3]))
            pred_mask_shape = interpolate_bilinear(pred_mask_shape)
            pred_uv_shape = interpolate_nearest(pred_uv_shape)

            if weight is not None:
                weight = weight[:, None, None, None] * has_dp.astype(weight.dtype).mean()
            else:
                weight = 1.0

            pred_mask_shape = ops.clip_by_value(pred_mask_shape, 0.0, 1.0)

            loss_mask = self.criterion_mask(pred_mask_shape, gt_mask_shape)
            gt_uv_weight = (gt_uv_shape.abs().max(axis=1, keepdims=True) > 0).astype(pred_dp.dtype)
            weight_ratio = gt_uv_weight.mean(axis=-1).mean(axis=-1)[:, :, None, None] + 1e-8
            gt_uv_weight = gt_uv_weight / weight_ratio

            loss_uv = self.criterion_uv(gt_uv_weight * pred_uv_shape, gt_uv_weight * gt_uv_shape)
            loss_uv = (loss_uv * weight).mean()

            return loss_mask, loss_uv
        return pred_dp.sum() * 0, pred_dp.sum() * 0

    def construct(self, *inputs, **kwargs):
        dtype = mindspore.float32
        has_dp, images, gt_dp_iuv, fit_joint_error = inputs
        pred_dp, dp_feature, codes = self.CNet(images)

        if self.adaptive_weight:
            ada_weight = self.error_adaptive_weight(fit_joint_error).astype(dtype)
        else:
            ada_weight = None

        #loss on dense pose result
        loss_dp_mask, loss_dp_uv = self.dp_loss(pred_dp, gt_dp_iuv, has_dp, ada_weight)
        loss_dp_mask = loss_dp_mask * self.lam_dp_mask
        loss_dp_uv = loss_dp_uv * self.lam_dp_uv

        loss_total = loss_dp_mask + loss_dp_uv

        return loss_total, pred_dp, dp_feature, codes
