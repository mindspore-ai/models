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


"""SalEdgeLoss define"""

import mindspore as ms
from mindspore import nn
from mindspore import Parameter


class TotalLoss(nn.Cell):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.BCELoss(reduction="mean")
        self.zero = ms.Tensor(0, dtype=ms.float32)
        # for log
        self.sal_loss = Parameter(default_input=0.0, requires_grad=False)
        self.edge_loss = Parameter(default_input=0.0, requires_grad=False)
        self.total_loss = Parameter(default_input=0.0, requires_grad=False)


    def construct(self, pres, gts, edges):

        loss_edg_5 = self.loss_fn1(pres[1], edges)
        loss_sal_5 = self.loss_fn2(pres[2], gts)
        loss_5 = loss_sal_5 + loss_edg_5
        loss_4 = self.loss_fn1(pres[3], edges) + self.loss_fn2(pres[4], gts)
        loss_3 = self.loss_fn1(pres[5], edges) + self.loss_fn2(pres[6], gts)
        loss_2 = self.loss_fn1(pres[7], edges) + self.loss_fn2(pres[8], gts)
        loss_1 = self.loss_fn1(pres[10], edges) + self.loss_fn2(pres[9], gts)
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        return loss


class WithLossCell(nn.Cell):
    """
    loss cell
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, gts, edges):
        """
        compute loss
        """
        pres = self.backbone(data)
        return self.loss_fn(pres, gts, edges)

    @property
    def backbone_network(self):
        return self.backbone
