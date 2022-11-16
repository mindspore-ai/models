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
""" all loss """
from mindspore import nn
from src.model.aux_ce_loss import Aux_CELoss_Cell
from src.model.affinity_loss import AffinityLoss


class All_Loss(nn.Cell):
    """ loss """
    def __init__(self):
        super(All_Loss, self).__init__()
        self.auxloss = Aux_CELoss_Cell()
        self.affinityloss = AffinityLoss()

    def construct(self, net_out, target):
        """ the process of calculate loss """
        predict, predict_aux, cpmap = net_out
        CE_loss = self.auxloss(predict, target)
        CE_loss_aux = self.auxloss(predict_aux, target)
        Affinityloss = self.affinityloss(cpmap, target)
        loss = CE_loss + (0.4 * CE_loss_aux) + Affinityloss
        return loss
