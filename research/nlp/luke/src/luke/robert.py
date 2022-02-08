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
"""roberta model"""
import math
import mindspore.numpy as mnp
import mindspore
from mindspore import nn, Parameter
from mindspore.nn import LayerNorm
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.common.initializer import initializer, TruncatedNormal


class TransposeForScores(nn.Cell):
    """transpose scores"""

    def __init__(self, num_attention_heads, attention_head_size):
        """init fun"""
        super(TransposeForScores, self).__init__()
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """construct fun"""
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return self.transpose(x, (0, 2, 1, 3))


class BertSelfAttention(nn.Cell):
    """bert self attention"""

    def __init__(self, config):
        """init fun"""
        super(BertSelfAttention, self).__init__()
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.key = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.value = nn.Dense(config.hidden_size, self.all_head_size).to_float(mindspore.float16)
        self.transpose_for_scores = TransposeForScores(
            num_attention_heads=self.num_attention_heads, attention_head_size=self.attention_head_size)
        self.dropout = nn.Dropout(1 - config.attention_probs_dropout_prob)
        self.matmul = ops.BatchMatMul()
        self.permute = ops.Transpose()
        self.transpose = ops.Transpose()
        self.view = ops.Reshape()
        self.softmax = ops.Softmax()
        self.sqrt_head_size = math.sqrt(self.attention_head_size)
        self.cast = ops.Cast()

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        """construct fun"""
        hidden_states = self.cast(hidden_states, mindspore.float16)
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = self.matmul(query_layer, self.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores / self.sqrt_head_size
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_probs = attention_probs.astype(mindspore.float16)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.permute(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertAttention(nn.Cell):
    """bert attention"""

    def __init__(self, config):
        """init fun"""
        super(BertAttention, self).__init__()
        # self.self 在mindspore会出现问题
        self.self1 = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
        self.view = P.Reshape()
        self.eq = ops.Equal()

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        """construct fun"""
        self_outputs = self.self1(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RobertaLMHead(nn.Cell):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size, weight_init=weight_init).to_float(
            mindspore.float16)
        self.layer_norm = LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(mindspore.float16)
        self.bias = Parameter(initializer('zero', config.vocab_size))
        self.gelu = GeLU()
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.matmul = P.MatMul(transpose_b=True)
        self.cast = ops.Cast()

    def construct(self, features, embedding_table):
        """construct fun"""
        features = self.cast(features, mindspore.float16)
        embedding_table = self.cast(embedding_table, mindspore.float16)
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        logits = self.matmul(x, embedding_table)
        logits = logits + self.bias

        return logits


class BertLayer(nn.Cell):
    """layer"""

    def __init__(self, config):
        """init fun"""
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        """construct fun"""
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Cell):
    """bert encoder"""

    def __init__(self, config):
        """init fun"""
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        """construct fun"""
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class RobertaEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        """init fun"""
        super(RobertaEmbeddings, self).__init__()
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(mindspore.float16)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.unsqueeze = ops.ExpandDims()
        self.cumsum = ops.CumSum()
        self.not_equal = ops.NotEqual()
        self.cast = ops.Cast()

    def construct(self, input_ids, token_type_ids):
        """construct fun"""
        mask = self.not_equal(input_ids, self.padding_idx).astype(mindspore.int32)
        incremental_indicies = self.cumsum(mask, 1) * mask
        position_ids = incremental_indicies + self.padding_idx
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.cast(embeddings, mindspore.float16)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        """init fun"""
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(mindspore.float16)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.unsqueeze = ops.ExpandDims()
        self.cast = ops.Cast()

    def construct(self, input_ids, token_type_ids):
        """construct fun"""
        input_shape = input_ids.shape

        seq_length = input_shape[1]
        position_ids = mnp.arange(seq_length, dtype=mindspore.int32)
        broadcast_to = ops.BroadcastTo(input_shape)
        position_ids = broadcast_to(self.unsqueeze(position_ids, 0))
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.cast(embeddings, mindspore.float16)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertOutput(nn.Cell):
    """bert output"""

    def __init__(self, config):
        """init fun"""
        super(BertOutput, self).__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size).to_float(mindspore.float16)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(mindspore.float16)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.cast = ops.Cast()

    def construct(self, hidden_states, input_tensor):
        """construct fun"""
        hidden_states = self.cast(hidden_states, mindspore.float16)
        input_tensor = self.cast(input_tensor, mindspore.float16)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GeLU(nn.Cell):
    """gelu layer"""

    def __init__(self):
        """init fun"""
        super(GeLU, self).__init__()
        self.div = P.Div()
        self.div_w = 1.4142135381698608
        self.erf = P.Erf()
        self.add = P.Add()
        self.add_bias = 1.0
        self.mul = P.Mul()
        self.mul_w = 0.5

    def construct(self, x):
        """construct function"""
        output = self.div(x, self.div_w)
        output = self.erf(output)
        output = self.add(output, self.add_bias)
        output = self.mul(x, output)
        output = self.mul(output, self.mul_w)
        return output


class BertIntermediate(nn.Cell):
    """bert intermdiate fun"""

    def __init__(self, config):
        """init fun"""
        super(BertIntermediate, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size).to_float(mindspore.float16)
        self.intermediate_act_fn = GeLU()
        self.cast = ops.Cast()

    def construct(self, hidden_states):
        """construct fun"""
        hidden_states = self.cast(hidden_states, mindspore.float16)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertSelfOutput(nn.Cell):
    """bert self output"""

    def __init__(self, config):
        """init fun"""
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(mindspore.float16)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(mindspore.float16)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.cast = ops.Cast()

    def construct(self, hidden_states, input_tensor):
        """construct fun"""
        hidden_states = self.cast(hidden_states, mindspore.float16)
        input_tensor = self.cast(input_tensor, mindspore.float16)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
