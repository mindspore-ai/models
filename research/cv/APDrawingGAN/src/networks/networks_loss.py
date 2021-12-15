# Copyright 2021Huawei Technologies Co., Ltd
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
"""networks loss"""

from mindspore import nn, ops
from mindspore import dtype as mstype


class GANLoss(nn.Cell):
    """GANLoss"""
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss(reduction='mean')

    def get_label_ops(self, target_is_real):
        if target_is_real:
            label_ops = ops.Ones()
        else:
            label_ops = ops.Zeros()
        return label_ops

    def construct(self, inputT, target_is_real):
        label_ops = self.get_label_ops(target_is_real)
        label = label_ops(inputT.shape, mstype.float32)
        loss = self.loss(inputT, label)
        return loss
