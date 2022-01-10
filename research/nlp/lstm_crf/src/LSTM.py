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
"""LSTM."""

from mindspore import nn
from mindspore.ops import operations as P


class Lstm(nn.Cell):
    """
    Stack multi-layers LSTM together.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 out_size,
                 weight,
                 num_layers=1,
                 batch_size=1,
                 dropout=0.0,
                 bidirectional=True):
        super(Lstm, self).__init__()
        # Mapp words to vectors
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size,
                                      use_one_hot=False,
                                      embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False

        self.perm = (1, 0, 2)
        self.trans = P.Transpose()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            has_bias=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        if bidirectional:
            self.fc = nn.Dense(hidden_size * 2, out_size)
        else:
            self.fc = nn.Dense(hidden_size, out_size)


    def construct(self, sequence_tensor):
        """the output of LSTM"""
        embeddings = self.embedding(sequence_tensor)
        embeddings = self.trans(embeddings, self.perm)
        lstm_out, _ = self.lstm(embeddings)
        if self.bidirectional:
            lstm_out = lstm_out.view(embeddings.shape[0], embeddings.shape[1], self.hidden_size*2)
        else:
            lstm_out = lstm_out.view(embeddings.shape[0], embeddings.shape[1], self.hidden_size)
        lstm_feats = self.fc(lstm_out)
        return lstm_feats
