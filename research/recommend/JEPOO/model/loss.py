# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import ops


def focal_loss(inputs, targets, alpha, gamma, weight):
    loss = ops.BinaryCrossEntropy(reduction='none')(inputs, targets, weight)
    weight[targets == 1] = float(alpha)
    loss_w = ops.BinaryCrossEntropy(reduction='none')(inputs, targets, weight)
    pt = ops.exp(-loss)
    weight_gamma = (1 - pt) ** gamma
    res_loss = (weight_gamma * loss_w).mean()
    return res_loss
