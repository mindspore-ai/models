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
"""MIFGSM"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class MIFGSM():
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0):
        self.model = model
        self.eps = Tensor(eps)
        self.steps = steps
        self.decay = Tensor(decay)
        self.alpha = Tensor(alpha)
        self._supported_mode = ['default', 'targeted']
        self.min_value = Tensor(0, mindspore.float32)
        self.max_value = Tensor(1, mindspore.float32)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.sign = ops.Sign()
        self.zeroslike = ops.ZerosLike()
        self.abs = ops.Abs()
        self.mean = ops.ReduceMean(keep_dims=True)

    def forward(self, images, labels):
        images = images.copy()
        labels = labels.copy()
        adv_images = images.copy()
        momentum = self.zeroslike(images)

        for _ in range(self.steps):
            net_with_criterion = nn.WithLossCell(self.model, self.loss)
            train_network = nn.ForwardValueAndGrad(net_with_criterion)
            # Update adversarial images
            grad = train_network(adv_images, labels)[1]
            grad = grad / self.mean(self.abs(grad), axis=(1, 2, 3))
            grad = grad + momentum*self.decay
            momentum = grad
            adv_images = adv_images + self.alpha*self.sign(grad)
            delta = ops.clip_by_value(adv_images-images, -self.eps, self.eps)
            adv_images = ops.clip_by_value(images+delta, self.min_value, self.max_value)

        return adv_images
