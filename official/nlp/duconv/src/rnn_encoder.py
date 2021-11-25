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
"""
knowledge encoder
"""
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from .rnns import GRU


class RNNEncoder(nn.Cell):
    """
    RNNEncoder
    """
    def __init__(self, input_size, hidden_size, bidirectional, dropout=0.0, embeddings=None, use_bridge=False):
        super().__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.bidirectional = bidirectional
        self.embeddings = embeddings
        self.hidden_size = hidden_size

        self.rnn = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        ).to_float(mstype.float16)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self.total_hidden_dim = hidden_size
            self.bridge = nn.Dense(self.total_hidden_dim,\
                                   self.total_hidden_dim, activation='relu').to_float(mstype.float16)
    def construct(self, inputs, seq_length):
        """
        construct
        """
        emb = self.embeddings(inputs)
        memory_bank, encoder_final = self.rnn(emb)

        if self.use_bridge:
            shape = encoder_final.shape
            encoder_final = self.bridge(encoder_final.view(-1, self.total_hidden_dim)).view(shape)
        if self.bidirectional:
            batch_size = encoder_final.shape[1]
            encoder_final = encoder_final.view(1, 2, batch_size, self.hidden_size) \
                .swapaxes(1, 2).view(1, batch_size, self.hidden_size * 2)
        return encoder_final, memory_bank
