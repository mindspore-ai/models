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

import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np

class AdditiveAngularMargin(nn.Cell):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity.
    scale: float
        The scale for cosine similarity.

    Returns
    -------
    predictions : Tensor.

    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.sqrt = ms.ops.Sqrt()
        self.pow = ms.ops.Pow()

    def construct(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : Tensor
        """
        cosine = outputs
        sine = self.sqrt(1.0 - self.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = np.where(cosine > 0, phi, cosine)
        else:
            phi = np.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs
