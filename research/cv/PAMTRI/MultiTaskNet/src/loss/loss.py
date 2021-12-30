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
"""loss"""
import mindspore.nn as nn

from .hard_mine_triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLabelSmooth

def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss

class netWithLossCell(nn.Cell):
    """newWithLoss"""
    def __init__(self, network, label_smooth, keyptaware, multitask, htri_only, lambda_xent,
                 lambda_htri, lambda_vcolor, lambda_vtype, margin, num_train_vids, num_train_vcolors, num_train_vtypes, batch_size):
        super(netWithLossCell, self).__init__()
        self.keyptaware = keyptaware
        self.multitask = multitask
        self.htri_only = htri_only
        self.lambda_xent = lambda_xent
        self.lambda_htri = lambda_htri
        self.lambda_vcolor = lambda_vcolor
        self.lambda_vtype = lambda_vtype
        self.network = network

        self.criterion_htri = TripletLoss(batch_size, margin=margin)
        if label_smooth:
            self.criterion_xent_vid = CrossEntropyLabelSmooth(num_classes=num_train_vids)
            self.criterion_xent_vcolor = CrossEntropyLabelSmooth(num_classes=num_train_vcolors)
            self.criterion_xent_vtype = CrossEntropyLabelSmooth(num_classes=num_train_vtypes)
        else:
            self.criterion_xent_vid = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            self.criterion_xent_vcolor = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            self.criterion_xent_vtype = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, img, vid, camid, vcolor, vtype, vkeypt):
        """netWithLoss construct"""
        output_vids, output_vcolors, output_vtypes, features = self.network(img, vkeypt)

        if self.htri_only:
            loss = self.criterion_htri(features, vid)
        else:
            xent_loss = self.criterion_xent_vid(output_vids, vid)
            htri_loss = self.criterion_htri(features, vid)

            loss = self.lambda_xent * xent_loss + self.lambda_htri * htri_loss

        if self.multitask:
            xent_loss_vcolor = self.criterion_xent_vcolor(output_vcolors, vcolor)
            xent_loss_vtype = self.criterion_xent_vtype(output_vtypes, vtype)

            loss += self.lambda_vcolor * xent_loss_vcolor + self.lambda_vtype * xent_loss_vtype

        return loss
