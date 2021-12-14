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

"""yolact training network wrapper."""

import mindspore.nn as nn

time_stamp_init = False
time_stamp_first = 0

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss_fn

    def construct(self, x, gt_bboxes, gt_label, crowd_boxes, gt_mask):

        # prediction dict: six tensor
        prediction = self._net(x)
        loss_B, loss_C, loss_M, loss_S, loss_I = self._loss(prediction,
                                                            gt_bboxes, gt_label, crowd_boxes, gt_mask)
        loss = loss_B + loss_C + loss_M + loss_S + loss_I
        return loss


    @property
    def network(self):
        """
        Get the network.

        Returns:
            Cell, return backbone network.
        """
        return self._net
