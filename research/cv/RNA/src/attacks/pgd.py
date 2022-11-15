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
"""PGD"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class PGD():
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, random_start=True):
        self.model = model
        self.eps = Tensor(eps)
        self.alpha = Tensor(alpha)
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.min_value = Tensor(0, mindspore.float32)
        self.max_value = Tensor(1, mindspore.float32)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.sign = ops.Sign()

    def forward(self, images, labels):
        images = images.copy()
        labels = labels.copy()
        adv_images = images.copy()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + ops.uniform(adv_images.shape, -self.eps, self.eps)
            adv_images = ops.clip_by_value(adv_images, self.min_value, self.max_value)

        for _ in range(self.steps):
            net_with_criterion = nn.WithLossCell(self.model, self.loss)
            train_network = nn.ForwardValueAndGrad(net_with_criterion)

            # Update adversarial images
            grad = train_network(adv_images, labels)[1]
            adv_images = adv_images + self.alpha*self.sign(grad)
            delta = ops.clip_by_value(adv_images-images, -self.eps, self.eps)
            adv_images = ops.clip_by_value(images+delta, self.min_value, self.max_value)

        return adv_images
