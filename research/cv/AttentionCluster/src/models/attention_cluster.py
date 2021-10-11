# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Attention Cluster structure"""
import mindspore
import mindspore.nn as nn
import mindspore.common.initializer as initializer
import numpy as np


class ShiftingAttention(nn.Cell):
    """shifting attention module"""
    def __init__(self, dim, n, fc):
        super(ShiftingAttention, self).__init__()
        self.dim = dim
        self.n_att = n

        layers = []
        if fc == 1:
            layers.append(nn.Conv1d(dim, n, 1, has_bias=True))
        else:
            layers.append(nn.Conv1d(dim, 128, 1, has_bias=True))
            layers.append(nn.Tanh())
            layers.append(nn.Conv1d(128, n, 1, has_bias=True))
        self.seq = nn.SequentialCell(layers)

        self.gnorm = np.sqrt(n)

        self.glrt = np.sqrt(1.0 / np.sqrt(n))
        self.w = mindspore.Parameter(mindspore.Tensor(shape=(self.n_att,),
                                                      init=initializer.Normal(0.0, self.glrt), dtype=mindspore.float32))
        self.b = mindspore.Parameter(mindspore.Tensor(shape=(self.n_att,),
                                                      init=initializer.Normal(0.0, self.glrt), dtype=mindspore.float32))

        self.norm = nn.Norm(axis=-1, keep_dims=True)

    def construct(self, x):
        """construct"""
        # x = (N, L, F)
        scores = self.seq(x.transpose(0, 2, 1))
        # scores = (N, C, L)
        weights = softmax_m1(scores)
        # weights = (N, C, L), sum(weights, -1) = 1

        outs = []
        expand_dims = mindspore.ops.ExpandDims()
        broadcast_to = mindspore.ops.BroadcastTo(shape=(x.shape[0], x.shape[-1]))
        for i in range(self.n_att):
            weight = weights[:, i, :]
            # weight = (N, L)
            weight = expand_dims(weight, -1).expand_as(x)
            # weight = (N, L, F)

            w = broadcast_to(expand_dims(self.w[i], 0))
            b = broadcast_to(expand_dims(self.b[i], 0))
            # center = (N, L, F)

            o = (x * weight).sum(axis=1) * w + b

            norm2 = self.norm(o).expand_as(o)
            o = o / norm2 / self.gnorm
            outs.append(o)
        concat = mindspore.ops.Concat(axis=-1)
        outputs = concat(outs)
        # outputs = (N, F*C)
        return outputs, weights

def softmax_m1(x):
    flat_x = x.view(-1, x.shape[-1])
    softmax = nn.Softmax()
    flat_y = softmax(flat_x)
    y = flat_y.view(*x.shape)
    return y


class AttentionCluster(nn.Cell):
    """Attention Cluster"""
    def __init__(self, fdims, natts, nclass, fc=1):
        super(AttentionCluster, self).__init__()
        self.feature_num = len(fdims)
        self.att = nn.CellList()
        self.inch = 0
        for input_dim, cluster_num in zip(fdims, natts):
            att = ShiftingAttention(input_dim, cluster_num, fc=fc)
            self.att.append(att)
            self.inch += input_dim*cluster_num

        self.drop = nn.Dropout(0.5)

        self.fc = nn.Dense(self.inch, nclass)

    def construct(self, x):
        """construct"""
        if self.feature_num < 2:
            out, _ = self.att[0](x)
        else:
            att_outs = []
            for i, feature in enumerate(x):
                att_out, _ = self.att[i](feature)
                att_outs.append(att_out)
            stack = mindspore.ops.Stack(axis=1)
            out = stack(att_outs)

        out = self.drop(out)
        out = self.fc(out)
        return out
