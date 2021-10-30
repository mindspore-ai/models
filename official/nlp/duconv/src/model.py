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
'''
Retrieval model
'''
from mindspore.common.initializer import TruncatedNormal
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as P
from .rnn_encoder import RNNEncoder
from .bert import BertModel


class Attention(nn.Cell):
    """
    Attention
    """
    def __init__(self, memory_size, hidden_size):
        super().__init__()
        self.bmm = P.BatchMatMul()
        self.concat = P.Concat(-1)
        self.softmax = nn.Softmax()
        self.linear_project = nn.SequentialCell([
            nn.Dense(hidden_size + memory_size, hidden_size).to_float(mstype.float16),
            nn.Tanh()
        ])

    def construct(self, query, memory):
        attn = self.bmm(query, memory.swapaxes(1, 2))
        weights = self.softmax(attn)
        weighted_memory = self.bmm(weights, memory)
        project_output = self.linear_project(self.concat((weighted_memory, query)))
        return project_output

class Retrieval(nn.Cell):
    """
    Retrieval model
    """
    def __init__(self, config, use_kn=False):
        super().__init__()
        self.bert = BertModel(config)
        self.use_kn = use_kn
        if self.use_kn:
            self.kn_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.memory = RNNEncoder(
                config.hidden_size,
                config.hidden_size,
                True,
                config.hidden_dropout_prob,
                self.kn_embeddings,
                True)
            self.attention = Attention(config.hidden_size, config.hidden_size)
        self.fc = nn.Dense(config.hidden_size, 2,
                           weight_init=TruncatedNormal(config.initializer_range)).to_float(mstype.float16)
        self.dropout = nn.Dropout(1-config.hidden_dropout_prob)

    def construct(self, input_ids, segment_ids, position_ids=None, kn_ids=None, seq_length=None):
        """
        construct
        """
        if len(seq_length.shape) != 1:
            seq_length = P.Squeeze(1)(seq_length)
        _, h_pooled = self.bert(input_ids, segment_ids, position_ids)
        h_pooled = P.ExpandDims()(h_pooled, 1)
        if self.use_kn:
            _, memory_bank = self.memory(kn_ids, seq_length)
            cls_feats = self.attention(h_pooled, memory_bank)
        else:
            cls_feats = h_pooled
        cls_feats = self.dropout(cls_feats.squeeze(1))
        logits = self.fc(cls_feats)
        return logits

class RetrievalWithLoss(nn.Cell):
    """
    RetrievalWithLoss
    """
    def __init__(self, config, use_kn):
        super().__init__()
        self.network = Retrieval(config, use_kn)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.squeeze = P.Squeeze(1)
    def construct(self, *inputs):
        out = self.network(*inputs[:-1])
        labels = self.squeeze(inputs[-1])
        return self.loss(out, labels)

class RetrievalWithSoftmax(nn.Cell):
    """
    RetrievalWithSoftmax
    """
    def __init__(self, config, use_kn):
        super().__init__()
        self.network = Retrieval(config, use_kn)
        self.softmax = nn.Softmax()

    def construct(self, *inputs):
        out = self.network(*inputs)
        out = self.softmax(out)
        return out
