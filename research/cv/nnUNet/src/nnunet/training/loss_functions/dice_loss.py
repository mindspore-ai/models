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

"""dice loss module"""

import mindspore.ops as ops
from mindspore import nn
from mindspore.nn.loss.loss import LossBase

from src.nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss, RobustCrossEntropyLoss2d
from src.nnunet.utilities.nd_softmax import softmax_helper


class SoftDiceLoss(nn.DiceLoss):
    """soft dice loss module"""

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., loss_type='3d'):
        super(SoftDiceLoss, self).__init__()
        self.mean = ops.ReduceMean()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.reshape = ops.Reshape()
        self.zeros = ops.Zeros()




class DC_and_CE_loss(LossBase):
    """Dice and cross entrophy loss"""
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):

        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        if soft_dice_kwargs["loss_type"] == '3d':
            self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        else:
            self.ce = RobustCrossEntropyLoss2d(**ce_kwargs)

        self.transpose = ops.Transpose()
        self.ignore_label = ignore_label
        self.reshape = ops.Reshape()

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def construct(self, net_output, target):
        """construct network"""
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        target = target
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
