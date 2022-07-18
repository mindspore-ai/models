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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.nn import LossBase
from mindspore.ops import Zeros
from mindspore.ops import ExpandDims, Concat, Squeeze
from src.utils.additional_algorithms import linear


class ProjectedAdaptiveLogSoftmaxLoss(LossBase):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, tie_projs=None,
                 keep_order=False):
        super(ProjectedAdaptiveLogSoftmaxLoss, self).__init__()
        self.squeeze_1 = Squeeze(1)
        self.gather = P.GatherD()
        self.zeros = Zeros()
        self.expandDims = ExpandDims()
        self.concat_0 = Concat(0)
        self.log_softmax_n_1 = nn.LogSoftmax()
        self.log_softmax_1 = nn.LogSoftmax(1)
        if tie_projs is None:
            tie_projs = [False]
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = ms.Parameter(self.zeros((self.n_clusters, self.d_embed), ms.float32))
            self.cluster_bias = ms.Parameter(self.zeros(self.n_clusters, ms.float32))

        self.out_layers = nn.CellList()
        parameters = []

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    parameters.append(
                        ms.Parameter(self.zeros((d_proj, d_embed), ms.float32))
                    )

            self.out_layers.append(nn.Dense(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)

                parameters.append(
                    ms.Parameter(self.zeros((d_proj, d_emb_i), ms.float32))
                )

                self.out_layers.append(nn.Dense(d_emb_i, r_idx - l_idx))

        self.out_projs = ms.ParameterTuple(parameters)
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj=None):
        if proj is None:
            logit = linear(hidden, weight, bias)
        else:
            proj_hid = linear(hidden, proj.T)
            logit = linear(proj_hid, weight, bias)
        return logit

    def construct(self, hidden, target):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        logit = self.out_layers[0](hidden)
        nll = self.squeeze_1(self.gather(-self.log_softmax_n_1(logit), 1, self.expandDims(target, 1)))
        return self.get_loss(nll)
