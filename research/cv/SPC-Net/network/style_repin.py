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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update


def calculate_mu_sig(x, eps=1e-6):
    mu = ops.ReduceMean(keep_dims=False)(x, (2, 3))
    var = x.var(axis=(2, 3))
    sig = ops.Sqrt()(var + eps)
    return mu, sig


class StyleRepresentation(nn.Cell):
    def __init__(
            self,
            num_prototype=2,
            channel_size=64,
            batch_size=4,
            gamma=0.9,
            dis_mode='was',
            channel_wise=False
        ):
        super(StyleRepresentation, self).__init__()
        self.num_prototype = num_prototype
        self.channel_size = channel_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.dis_mode = dis_mode
        self.channel_wise = channel_wise
        style_init = Tensor(np.zeros((self.num_prototype, self.channel_size)), ms.float32)
        self.style_mu = ms.Parameter(style_init, requires_grad=True)
        self.style_sig = ms.Parameter(style_init, requires_grad=True)

    def was_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)
        distance = ops.pow((cur_mu - proto_mu), 2) + ops.pow(cur_sig, 2) + \
            ops.pow(proto_sig, 2) - 2 * cur_sig * proto_sig
        return distance

    def construct(self, fea):
        batch = fea.shape[0]
        proto_mu = self.style_mu
        proto_sig = self.style_sig

        cur_mu, cur_sig = calculate_mu_sig(fea)
        if self.dis_mode == 'was':
            distance = self.was_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)
        else: # abs kl others
            raise NotImplementedError('No this distance mode!')

        if not self.channel_wise:
            distance = ops.ReduceMean(keep_dims=False)(distance, 2)
        alpha = 1.0 / (1.0 + distance)
        alpha = ops.Softmax(axis=1)(alpha)

        if not self.channel_wise:
            mixed_mu = ops.matmul(alpha, proto_mu)
            mixed_sig = ops.matmul(alpha, proto_sig)
        else:
            raise NotImplementedError('No this distance mode!')

        fea = ((fea - cur_mu[:, :, None, None]) / cur_sig[:, :, None, None]) * \
            mixed_sig[:, :, None, None] + mixed_mu[:, :, None, None]

        return fea
