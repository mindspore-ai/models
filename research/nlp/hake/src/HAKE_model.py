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
"""hake model"""

import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.ops import Gather, ExpandDims, Split, Abs, CumSum, Sin, Select, Cast, Print
import numpy as np
from src.utils import uniform_weight, rel_init


class HAKE_GRAPH(nn.Cell):
    """
    HAKE base model.

    Args:
        num_entity (int): entity nums
        num_relation (int): relation nums
        hidden_dim (int): embedding dimensions
        gamma (float): global hyper-parameters.
        modulus_weight (float): global hyper-parameters.
        phase_weight (float): global hyper-parameters.
    """

    def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE_GRAPH, self).__init__()

        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.gamma = gamma

        self.embedding_range = (self.gamma + self.epsilon) / hidden_dim

        self.entity_embedding = Parameter(uniform_weight(self.embedding_range, (num_entity, hidden_dim * 2)))
        self.relation_embedding = Parameter(rel_init(self.embedding_range, (num_relation, hidden_dim * 3), hidden_dim))

        self.phase_weight = Parameter(Tensor([[phase_weight * self.embedding_range]], dtype=mindspore.dtype.float32))
        self.modulus_weight = Parameter(Tensor([[modulus_weight]], dtype=mindspore.dtype.float32))

        self.gather = Gather()
        self.expand_dim = ExpandDims()
        self.split2 = Split(2, 2)
        self.split3 = Split(2, 3)
        self.abs = Abs()
        self.cumsum = CumSum()
        self.sin = Sin()
        self.select = Select()
        self.norm = nn.Norm(axis=2)
        self.cast = Cast()

        self.print = Print()

        self.pi = np.pi

    def score(self, mod_head, mod_relation, mod_tail, bias_relation, phase_score):
        """ calculate score base function """
        mod_relation = self.abs(mod_relation)
        bias_relation = bias_relation.clip(xmin=None, xmax=1)

        indicator = (bias_relation < -mod_relation)

        bias_relation_new = self.select(indicator, -mod_relation, bias_relation)

        r_score = mod_head * (mod_relation + bias_relation_new) - mod_tail * (1 - bias_relation_new)
        phase_score = self.abs(self.sin(phase_score / 2)).sum(axis=2) * self.phase_weight
        r_score = self.norm(r_score) * self.modulus_weight

        return self.cast(self.gamma - (phase_score + r_score), mindspore.float32)

    def score_head(self, head, rel, tail):
        """ calculate the triple score of head negative triple"""
        phase_head, mod_head = self.split2(head)
        phase_relation, mod_relation, bias_relation = self.split3(rel)
        phase_tail, mod_tail = self.split2(tail)

        phase_head = phase_head / (self.embedding_range / self.pi)
        phase_relation = phase_relation / (self.embedding_range / self.pi)
        phase_tail = phase_tail / (self.embedding_range / self.pi)

        phase_score = phase_head + (phase_relation - phase_tail)
        return self.score(mod_head, mod_relation, mod_tail, bias_relation, phase_score)

    def score_tail_single(self, head, rel, tail):
        """ calculate the triple score of tail negative triple or positive triple"""
        phase_head, mod_head = self.split2(head)
        phase_relation, mod_relation, bias_relation = self.split3(rel)
        phase_tail, mod_tail = self.split2(tail)

        phase_head = phase_head / (self.embedding_range / self.pi)
        phase_relation = phase_relation / (self.embedding_range / self.pi)
        phase_tail = phase_tail / (self.embedding_range / self.pi)

        phase_score = (phase_head + phase_relation) - phase_tail

        return self.score(mod_head, mod_relation, mod_tail, bias_relation, phase_score)

    def construct_head(self, sample):
        """
        calculate the score of head negative training triple
        Args:
            sample: different format for different batch types. tail negative: (positive_sample, negative_sample)
                positive_sample is the tensor with shape [batch_size, 3]
                negative_sample is the tensor with shape [batch_size, negative_sample_size]

        Returns: Tensor, triple score

        """
        tail_part, head_part = sample
        batch_size, negative_sample_size = head_part.shape[0], head_part.shape[1]

        # shape [batch_size, negative_sample_size, embedding_size]
        head = self.gather(self.entity_embedding, head_part.view(-1), 0).view(batch_size, negative_sample_size, -1)

        relation = self.gather(self.relation_embedding, tail_part[:, 1], 0)
        relation = self.expand_dim(relation, 1)

        tail = self.gather(self.entity_embedding, tail_part[:, 2], 0)
        tail = self.expand_dim(tail, 1)

        return self.score_head(head, relation, tail)

    def construct_tail(self, sample):
        """
        calculate the score of tail negative training triple
        Args:
            sample: different format for different batch types. tail negative: (positive_sample, negative_sample)
                positive_sample is the tensor with shape [batch_size, 3]
                negative_sample is the tensor with shape [batch_size, negative_sample_size]

        Returns: Tensor, triple score

        """
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.shape[0], tail_part.shape[1]

        head = self.gather(self.entity_embedding, head_part[:, 0], 0)
        head = self.expand_dim(head, 1)

        relation = self.gather(self.relation_embedding, head_part[:, 1], 0)
        relation = self.expand_dim(relation, 1)

        # shape [batch_size, negative_sample_size, embedding_size]
        tail = self.gather(self.entity_embedding, tail_part.view(-1), 0).view(batch_size, negative_sample_size, -1)

        return self.score_tail_single(head, relation, tail)

    def construct_single(self, sample):
        """
        calculate the score of positive training triple
        Args:
            sample: different format for different batch types. SINGLE: tensor with shape [batch_size, 3]

        Returns: Tensor, triple score

        """
        head = self.gather(self.entity_embedding, sample[:, 0], 0)
        head = self.expand_dim(head, 1)

        relation = self.gather(self.relation_embedding, sample[:, 1], 0)
        relation = self.expand_dim(relation, 1)

        tail = self.gather(self.entity_embedding, sample[:, 2], 0)
        tail = self.expand_dim(tail, 1)

        return self.score_tail_single(head, relation, tail)

    def construct(self):
        pass
