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
"""Loss function"""

import mindspore
from mindspore import Tensor
from mindspore import nn
from mindspore import ops

from src.box_utils import log_sum_exp


class MultiBoxLoss(nn.Cell):
    """SSD Weighted Loss Function"""

    def __init__(self,
                 num_classes,
                 neg_pos):
        super().__init__()
        self.num_classes = num_classes
        self.negpos_ratio = neg_pos

        self.zero_float = Tensor(0, dtype=mindspore.float32)
        self.one_float = Tensor(1, dtype=mindspore.float32)

    def construct(self, predictions, loc_t, conf_t):
        """Forward pass"""

        loc, conf = predictions
        num = loc.shape[0]

        unsqueeze = ops.ExpandDims()
        pos = ops.repeat_elements(unsqueeze(conf_t, conf_t.ndim), rep=loc.shape[-1], axis=conf_t.ndim)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        loc_p = ops.Select()(pos > 0, loc, ops.ZerosLike()(loc))
        loc_t = ops.Select()(pos > 0, loc_t, ops.ZerosLike()(loc))
        loss_l = ops.SmoothL1Loss()(loc_p, loc_t).sum()

        # Compute max conf across batch for hard negative mining
        batch_conf = conf.view(-1, self.num_classes)
        loss_cls = log_sum_exp(batch_conf) - ops.GatherD()(batch_conf, 1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_cls = ops.Select()(conf_t.view(-1, 1) > 0, ops.ZerosLike()(loss_cls), loss_cls)  # filter out pos boxes
        loss_cls = loss_cls.view(num, -1)
        _, loss_idx = ops.Sort(axis=1, descending=True)(loss_cls)
        _, idx_rank = ops.Sort(axis=1)(loss_idx.astype('float32'))
        pos_number = (conf_t > 0).astype('int32').sum(1, keepdims=True)
        negpos_numpos = (self.negpos_ratio * pos_number).astype('float32')
        num_neg = ops.clip_by_value(negpos_numpos, clip_value_min=negpos_numpos.min(), clip_value_max=pos.shape[1] - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = unsqueeze((conf_t > 0).astype('int32'), 2).expand_as(conf)
        neg_idx = unsqueeze(neg.astype('int32'), 2).expand_as(conf)
        conf_p = ops.Select()(
            ops.Greater()(pos_idx + neg_idx, 0),
            conf,
            ops.ZerosLike()(conf),
        ).view(-1, self.num_classes)
        valid_targets = ops.Greater()((conf_t > 0).astype('int32') + neg.astype('int32'), 0)
        targets_weighted = ops.Select()(
            valid_targets,
            conf_t,
            ops.ZerosLike()(conf_t),
        ).view(-1).astype('int32')
        depth, on_value, off_value = self.num_classes, self.one_float, self.zero_float
        loss_cls, _ = ops.SoftmaxCrossEntropyWithLogits()(
            conf_p,
            ops.OneHot()(targets_weighted, depth, on_value, off_value),
        )
        loss_cls = ops.Select()(valid_targets.view(-1), loss_cls, ops.ZerosLike()(loss_cls))
        loss_cls = loss_cls.sum()

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = ops.Maximum()(pos_number.sum(), self.one_float)
        loss_l /= N
        loss_cls /= N
        return loss_l + loss_cls
