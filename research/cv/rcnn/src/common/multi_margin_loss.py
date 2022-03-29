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
"""
multi_margin_loss, will be used in training svm.
"""
from typing import Optional
import mindspore
from mindspore import Tensor
from mindspore import nn
from mindspore import ops


class MultiMarginLoss(nn.Cell):
    """
    MultiMarginLoss
    """

    def __init__(self, class_num, margin: float = 1., weight: Optional[Tensor] = None) -> None:
        super(MultiMarginLoss, self).__init__()
        assert weight is None or weight.dim() == 1
        self.weight = weight
        self.margin = margin
        self.on_value, self.off_value = Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        self.op_sum = ops.ReduceSum(keep_dims=True)
        self.onehot = nn.OneHot(depth=class_num, axis=-1)
        self.relu = nn.ReLU()

    def construct(self, input_: Tensor, target: Tensor) -> Tensor:
        """

        :param input_: input
        :param target: target
        :return: loss
        """
        target = self.onehot(target)
        loss = self.margin - self.op_sum(input_ * target, 1) + input_ - target
        if self.weight is not None:
            loss = loss * self.op_sum(target * self.weight, 1)
        return self.relu(loss).mean()
