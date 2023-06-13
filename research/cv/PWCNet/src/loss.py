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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P


def _downsample2d_as(inputs, target_as):
    _, _, h1, _ = P.Shape()(target_as)
    _, _, h2, _ = P.Shape()(inputs)
    resize = h2 // h1
    return nn.AvgPool2d(1, stride=(resize, resize))(inputs) * (1.0 / resize)


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return ops.norm(residual, dim=1, keepdim=True)


class MultiScaleEPE_PWC(LossBase):
    """define loss function"""

    def __init__(self):
        super(MultiScaleEPE_PWC, self).__init__()
        self.shape = P.Shape()
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def construct(self, outputs, target, training=True):
        N, _, _, _ = self.shape(target)
        if training:
            # div_flow trick
            target = 0.05 * target
            total_loss = 0
            loss_ii = _elementwise_epe(outputs[0], _downsample2d_as(target, outputs[0])).sum()
            total_loss = total_loss + self._weights[0] * loss_ii
            loss_ii = _elementwise_epe(outputs[1], _downsample2d_as(target, outputs[1])).sum()
            total_loss = total_loss + self._weights[1] * loss_ii
            loss_ii = _elementwise_epe(outputs[2], _downsample2d_as(target, outputs[2])).sum()
            total_loss = total_loss + self._weights[2] * loss_ii
            loss_ii = _elementwise_epe(outputs[3], _downsample2d_as(target, outputs[3])).sum()
            total_loss = total_loss + self._weights[3] * loss_ii
            loss_ii = _elementwise_epe(outputs[4], _downsample2d_as(target, outputs[4])).sum()
            total_loss = total_loss + self._weights[4] * loss_ii
            total_loss = total_loss / N

        else:
            epe = _elementwise_epe(outputs, target)
            total_loss = epe.mean()

        return total_loss
