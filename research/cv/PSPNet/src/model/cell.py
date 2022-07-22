# Copyright 2021 Huawei Technologies Co., Ltd
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
""" PSPNet loss function """
from mindspore import nn
from src.utils.aux_loss import SoftmaxCrossEntropyLoss


class Aux_CELoss_Cell(nn.Cell):
    """ loss """
    def __init__(self, num_classes=21, ignore_label=255):
        super(Aux_CELoss_Cell, self).__init__()
        self.num_classes = num_classes
        self.loss = SoftmaxCrossEntropyLoss(self.num_classes, ignore_label)

    def construct(self, net_out, target):
        """ the process of calculate loss """
        if len(net_out) == 2:
            predict_aux, predict = net_out
            CE_loss = self.loss(predict, target)
            CE_loss_aux = self.loss(predict_aux, target)
            loss = CE_loss + (0.4 * CE_loss_aux)
            return loss
        return self.loss(net_out, target)
