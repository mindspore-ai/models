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

from mindspore import nn, Tensor, ops
from mindspore import dtype as mstype
from mindspore import numpy as mnp
from src.config import cfg


class MultiBoxLoss(nn.Cell):
    """SSD Weighted Loss Function
    """
    def __init__(self, use_head_loss=False):
        super(MultiBoxLoss, self).__init__()
        self.use_head_loss = use_head_loss
        self.num_classes = cfg.NUM_CLASSES
        self.negpos_ratio = cfg.NEG_POS_RATIOS
        self.cast = ops.Cast()
        self.sum = ops.ReduceSum()
        self.loc_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.sort_descending = ops.Sort(descending=True)
        self.stack = ops.Stack(axis=1)
        self.unsqueeze = ops.ExpandDims()
        self.gather = ops.GatherNd()

    def construct(self, predictions, targets):
        """Multibox Loss"""
        if self.use_head_loss:
            _, _, loc_data, conf_data = predictions
        else:
            loc_data, conf_data, _, _ = predictions

        loc_t, conf_t = targets
        loc_data = self.cast(loc_data, mstype.float32)
        conf_data = self.cast(conf_data, mstype.float32)
        loc_t = self.cast(loc_t, mstype.float32)
        conf_t = self.cast(conf_t, mstype.int32)

        batch_size, box_num, _ = conf_data.shape

        mask = self.cast(conf_t > 0, mstype.float32)
        pos_num = self.sum(mask, 1)

        loc_loss = self.sum(self.loc_loss(loc_data, loc_t), 2)
        loc_loss = self.sum(mask * loc_loss)

        # Hard Negative Mining
        con = self.cls_loss(conf_data.view(-1, self.num_classes), conf_t.view(-1))
        con = con.view(batch_size, -1)

        con_neg = con * (1 - mask)
        value, _ = self.sort_descending(con_neg)
        neg_num = self.cast(ops.minimum(self.negpos_ratio * pos_num, box_num), mstype.int32)
        batch_iter = Tensor(mnp.arange(batch_size), dtype=mstype.int32)
        neg_index = self.stack((batch_iter, neg_num))
        min_neg_score = self.unsqueeze(self.gather(value, neg_index), 1)
        neg_mask = self.cast(con_neg > min_neg_score, mstype.float32)
        all_mask = mask + neg_mask
        all_mask = ops.stop_gradient(all_mask)

        cls_loss = self.sum(con * all_mask)

        N = self.sum(pos_num)
        N = ops.maximum(self.cast(N, mstype.float32), 0.25)

        loc_loss /= N
        cls_loss /= N

        return loc_loss, cls_loss
