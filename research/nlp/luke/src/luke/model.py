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
"""luke model"""
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.ops as ops

from src.luke.robert import BertEncoder, BertSelfOutput, BertIntermediate, BertOutput, TransposeForScores, \
    RobertaEmbeddings
from src.luke.config import LukeConfig


class EntityEmbeddings(nn.Cell):
    """entity embedding"""

    def __init__(self, config):
        """init fun"""
        super(EntityEmbeddings, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0,
                                              embedding_table=TruncatedNormal(config.initializer_range))
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Dense(config.entity_emb_size, config.hidden_size,
                                                   weight_init=TruncatedNormal(config.initializer_range),
                                                   has_bias=False).to_float(mindspore.float16)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                embedding_table=TruncatedNormal(config.initializer_range))
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size,
                                                  embedding_table=TruncatedNormal(config.initializer_range))
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(mindspore.float16)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.zerosLike = ops.ZerosLike()
        self.unsqueeze = ops.ExpandDims()
        self.sum = ops.ReduceSum()
        self.min1 = Tensor(0, mindspore.int32)
        self.max1 = Tensor(100000, mindspore.int32)
        self.min2 = Tensor(1e-7, mindspore.float32)
        self.max2 = Tensor(100000, mindspore.float32)
        self.clamp = C.clip_by_value
        self.entity_emb_size = config.entity_emb_size
        self.hidden_size = config.hidden_size
        self.cast = ops.Cast()

    def construct(self, entity_ids, position_ids, token_type_ids=None):
        """construct fun"""
        if token_type_ids is None:
            token_type_ids = self.zerosLike(entity_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)
        entity_embeddings = self.cast(entity_embeddings, mindspore.float16)
        if self.entity_emb_size != self.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)
        position_embeddings = self.position_embeddings(self.clamp(position_ids, self.min1, self.max1))
        position_embedding_mask = self.unsqueeze((position_ids != -1).astype(mindspore.float32), -1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = self.sum(position_embeddings, -2)
        position_embeddings = position_embeddings / self.clamp(self.sum(position_embedding_mask, -2),
                                                               self.min2, self.max2)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LukeModel(nn.Cell):
    """luke model"""

    def __init__(self, config: LukeConfig):
        """init fun"""
        super(LukeModel, self).__init__()

        self.config = config

        self.encoder = BertEncoder(config)

        self.pooler = nn.Dense(config.hidden_size, config.hidden_size, activation="tanh",
                               weight_init='Uniform', bias_init='Uniform').to_float(mindspore.float16)

        self.embeddings = RobertaEmbeddings(config)
        self.embeddings.token_type_embeddings.requires_grad = False
        self.entity_embeddings = EntityEmbeddings(config)
        self.cat = ops.Concat(axis=1)
        self.unsqueeze = ops.ExpandDims()
        self._compute_extended_attention_mask = ComputeExtendedAttentionMask()

        self.none_layer = [None] * self.config.num_hidden_layers

    def construct(self,
                  word_ids, word_segment_ids, word_attention_mask, entity_ids=None, entity_position_ids=None,
                  entity_segment_ids=None, entity_attention_mask=None):
        """construct fun"""
        word_seq_size = ops.shape(word_ids)[1]

        embedding_output = self.embeddings(word_ids, word_segment_ids)

        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        if entity_ids is not None:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
            embedding_output = self.cat((embedding_output, entity_embedding_output))

        encoder_outputs = self.encoder(embedding_output, attention_mask, self.none_layer)
        sequence_output = encoder_outputs[0]
        word_sequence_output = sequence_output[:, :word_seq_size, :]
        pooled_output = self.pooler(sequence_output)

        if entity_ids is not None:
            entity_sequence_output = sequence_output[:, word_seq_size:, :]
            return (word_sequence_output, entity_sequence_output, pooled_output,) + encoder_outputs[1:]
        return (word_sequence_output, pooled_output,) + encoder_outputs[1:]


class ComputeExtendedAttentionMask(nn.Cell):
    """compute extended attention mask"""

    def __init__(self):
        """init fun"""
        super(ComputeExtendedAttentionMask, self).__init__()
        self.cat = ops.Concat(axis=1)
        self.unsqueeze = ops.ExpandDims()

    def construct(self, word_attention_mask, entity_attention_mask):
        """construct fun"""
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = self.cat((attention_mask, entity_attention_mask))
        extended_attention_mask = self.unsqueeze(self.unsqueeze(attention_mask, 1), 2)
        extended_attention_mask = extended_attention_mask.astype(mindspore.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class LukeEntityAwareAttentionModel(nn.Cell):
    """LukeEntityAwareAttentionModel"""

    def __init__(self, config):
        """init fun"""
        super(LukeEntityAwareAttentionModel, self).__init__(config)
        self.encoder = EntityAwareEncoder(config)
        self._compute_extended_attention_mask = ComputeExtendedAttentionMask()
        self.entity_embeddings = EntityEmbeddings(config)
        self.embeddings = RobertaEmbeddings(config)
        self.embeddings.token_type_embeddings.requires_grad = False
        self.cast = ops.Cast()

    def construct(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                  entity_segment_ids, entity_attention_mask):
        """construct fun"""
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        output = self.encoder(word_embeddings, entity_embeddings, attention_mask)
        return output


class EntityAwareSelfAttention(nn.Cell):
    """entity self attention"""

    def __init__(self, config):
        """init fun"""
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.key = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.value = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.w2e_query = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.e2w_query = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.e2e_query = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.dropout = nn.Dropout(1 - config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax()
        self.permute = ops.Transpose()
        self.cat1 = ops.Concat(1)
        self.cat2 = ops.Concat(2)
        self.cat3 = ops.Concat(3)
        self.sqrt = ops.Sqrt()
        self.split2 = ops.Split(2, 2)
        self.split1 = ops.Split(1, 2)
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.matmul = P.BatchMatMul()
        self.transpose_for_scores = TransposeForScores(
            num_attention_heads=self.num_attention_heads, attention_head_size=self.attention_head_size)
        self.x1 = mindspore.Tensor(self.attention_head_size, mindspore.float32)
        self.cast = ops.Cast()

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """construct fun"""
        word_size = ops.shape(word_hidden_states)[1]
        word_hidden_states = self.cast(word_hidden_states, mindspore.float16)
        entity_hidden_states = self.cast(entity_hidden_states, mindspore.float16)
        t1 = self.query(word_hidden_states)
        t2 = self.w2e_query(word_hidden_states)
        t3 = self.e2w_query(entity_hidden_states)
        t4 = self.e2e_query(entity_hidden_states)
        w2w_query_layer = self.transpose_for_scores(t1)
        w2e_query_layer = self.transpose_for_scores(t2)
        e2w_query_layer = self.transpose_for_scores(t3)
        e2e_query_layer = self.transpose_for_scores(t4)
        word_entity_state = self.cat1((word_hidden_states, entity_hidden_states))
        t5 = self.key(word_entity_state)
        key_layer = self.transpose_for_scores(t5)

        w2w_key_layer, w2e_key_layer = self.split2(key_layer)
        e2w_key_layer, e2e_key_layer = self.split2(key_layer)
        t6 = self.permute(w2w_key_layer, (0, 1, 3, 2))
        t7 = self.permute(w2e_key_layer, (0, 1, 3, 2))
        t8 = self.permute(e2w_key_layer, (0, 1, 3, 2))
        t9 = self.permute(e2e_key_layer, (0, 1, 3, 2))
        w2w_attention_scores = self.matmul(w2w_query_layer, t6)
        w2e_attention_scores = self.matmul(w2e_query_layer, t7)
        e2w_attention_scores = self.matmul(e2w_query_layer, t8)
        e2e_attention_scores = self.matmul(e2e_query_layer, t9)

        word_attention_scores = self.cat3([w2w_attention_scores, w2e_attention_scores])
        entity_attention_scores = self.cat3([e2w_attention_scores, e2e_attention_scores])
        attention_scores = self.cat2([word_attention_scores, entity_attention_scores])

        attention_scores = attention_scores / self.sqrt(self.x1)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        word_entity_state = self.cat1((word_hidden_states, entity_hidden_states))
        t10 = self.value(word_entity_state)
        value_layer = self.transpose_for_scores(t10)
        attention_probs = attention_probs.astype(mindspore.float16)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.permute(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = ops.shape(context_layer)[:-2] + (self.all_head_size,)
        context_layer = ops.reshape(context_layer, new_context_layer_shape)
        word_self_output = context_layer[:, :word_size, :]
        entity_self_output = context_layer[:, word_size:, :]
        return word_self_output, entity_self_output


class EntityAwareAttention(nn.Cell):
    """entity attention"""

    def __init__(self, config):
        """init fun"""
        super(EntityAwareAttention, self).__init__()
        self.self1 = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

        self.cat1 = ops.Concat(1)
        self.cast = ops.Cast()

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """construct fun"""
        word_self_output, entity_self_output = self.self1(word_hidden_states, entity_hidden_states, attention_mask)
        word_hidden_states = self.cast(word_hidden_states, mindspore.float16)
        entity_hidden_states = self.cast(entity_hidden_states, mindspore.float16)
        hidden_states = self.cat1([word_hidden_states, entity_hidden_states])
        self_output = self.cat1([word_self_output, entity_self_output])
        output = self.output(self_output, hidden_states)
        out_len = ops.shape(word_hidden_states)[1]
        return output[:, :out_len, :], output[:, out_len:, :]


class EntityAwareLayer(nn.Cell):
    """entity layer"""

    def __init__(self, config):
        """init fun"""
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)

        self.output = BertOutput(config)
        self.cat1 = ops.Concat(1)

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """construct fun"""
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask)
        attention_output = self.cat1((word_attention_output, entity_attention_output))
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        out_len = ops.shape(word_hidden_states)[1]
        return layer_output[:, : out_len, :], layer_output[:, out_len:, :]


class EntityAwareEncoder(nn.Cell):
    """entity encoder"""

    def __init__(self, config):
        """init fun"""
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.CellList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """construct"""
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask)
        return word_hidden_states, entity_hidden_states
