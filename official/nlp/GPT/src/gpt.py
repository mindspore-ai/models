# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""GPT model"""

import numpy as np
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._transformer.layers import _LayerNorm
from mindspore.parallel._transformer.transformer import AttentionMask, TransformerEncoder


class EmbeddingLookup(nn.Cell):
    """
    The embedding lookup table for vocabulary

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """
    def __init__(self, config):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.embedding_table = Parameter(initializer(TruncatedNormal(0.02), [self.vocab_size, self.embedding_size]))
        self.gather = P.Gather()
        self.shape = (-1, config.seq_length, config.embedding_size)
    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table.value()


class GPT_Model(nn.Cell):
    """
    The backbone of GPT network

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map

    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """
    def __init__(self, config):
        super(GPT_Model, self).__init__()
        self.get_attention_mask = AttentionMask(seq_length=config.seq_length)
        self.word_embedding = EmbeddingLookup(config)
        self.position_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                               embedding_table=TruncatedNormal(0.02))
        self.blocks = nn.CellList()
        self.encoder = TransformerEncoder(batch_size=config.batch_size,
                                          num_layers=config.num_layers,
                                          hidden_size=config.embedding_size,
                                          ffn_hidden_size=config.embedding_size * 4,
                                          seq_length=config.seq_length,
                                          num_heads=config.num_heads,)
        self.layernorm = _LayerNorm((config.embedding_size,)).to_float(config.compute_dtype)
        self.use_past = config.use_past
        self.past = tuple([None]*config.num_layers)
        self.num_layers = config.num_layers

    def construct(self, input_ids, input_mask, layer_past=None):
        """GPT model"""
        if not self.use_past:
            layer_past = self.past

        input_embedding, embedding_table = self.word_embedding(input_ids)

        batch_size, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (batch_size, 1))


        position_embedding = self.position_embedding(input_position)
        hidden_states = input_embedding + position_embedding

        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.get_attention_mask(input_mask)

        hidden_states, present_layer = self.encoder(hidden_states, attention_mask)
        output_state = self.layernorm(hidden_states)
        return output_state, present_layer, embedding_table

class GPT_Head(nn.Cell):
    """
    Head for GPT to get the logits of each token in the vocab

    Args:
        config(GPTConfig): the config of network

    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary

    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """
    def __init__(self, config):
        super(GPT_Head, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.embedding_size))
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits

class GPT(nn.Cell):
    """
    The GPT network consisting of two parts the backbone and the head

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map

    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, config):
        super(GPT, self).__init__()
        self.backbone = GPT_Model(config)
        self.head = GPT_Head(config)

    def construct(self, input_ids, input_mask, past=None):
        output_states, _, embedding_table = self.backbone(input_ids, input_mask, past)
        logits = self.head(output_states, embedding_table)
        return logits

class GPTWithLoss(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token

    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map

    Returns:
        output: Tensor, the loss of the network
    """
    def __init__(self, network, loss, eos_token=50256):
        super(GPTWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token

    def construct(self, input_ids, past=None):
        tokens = input_ids[:, :-1]
        input_mask = F.cast(F.not_equal(tokens, self.eos_token), mstype.float32)
        logits = self.network(tokens, input_mask, past)
        labels = input_ids[:, 1:]
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output

class EvalNet(nn.Cell):
    """
    GPT evaluation net

    Args:
        backbone: backbone network of GPT2/3
        generate: enable generate mode

    Inputs:
        input_ids: the tokenized inpus

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """
    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.Argmax()
        self.generate = generate
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask):
        """evaluation net"""
        input_mask = self.cast(input_mask, mstype.float32)
        logits = self.backbone(input_ids, input_mask)
        outputs = None
        if self.generate:
            outputs = nn.LogSoftmax()(logits)
            outputs = F.tensor_pow(np.e, outputs)
        else:
            outputs = self.argmax(logits)
        return outputs
