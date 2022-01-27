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

import mindspore
import mindspore.numpy as mnp
from mindspore import nn

EPSILON = 1e-7

class SparseCategoricalCrossentropy(nn.SoftmaxCrossEntropyWithLogits):
    """ Implementation of sparce categorical crossentropy loss """
    def __init__(self, reduction='none'):
        super().__init__(sparse=True, reduction=reduction)


    def construct(self, probs, labels):
        # NOTE: flatten to calculate loss in time dim independently
        probs = mnp.reshape(probs, (-1, probs.shape[-1]))
        labels = mnp.reshape(labels, (-1,))

        probs = probs.astype(mindspore.float32)
        probs = probs + EPSILON
        logits = mnp.log(probs)
        loss = super().construct(logits, labels)
        return loss
