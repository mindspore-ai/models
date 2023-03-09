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
"""roberta model"""

import copy
import math
from typing import Tuple
import numpy as np
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore import ops
import mindspore as ms
from mindspore import log as logger
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from src.utils import generate_arange_tensor

ACT2FN = {"gelu": ops.GeLU(), "relu": ops.ReLU(),
          "swish": ops.HSwish(), "mish": ops.Mish()}


class RobertaGenerationConfig:
    """
        Configuration for `RobertaModel`.

        Args:
            seq_length (int): Length of input sequence. Default: 128.
            vocab_size (int): The shape of each embedding vector. Default: 32000.
            hidden_size (int): Size of the Roberta encoder layers. Default: 768.
            num_hidden_layers (int): Number of hidden layers in the RobertaTransformer encoder
                               cell. Default: 12.
            num_attention_heads (int): Number of attention heads in the RobertaTransformer
                                 encoder cell. Default: 12.
            intermediate_size (int): Size of intermediate layer in the RobertaTransformer
                               encoder cell. Default: 3072.
            hidden_act (str): Activation function used in the RobertaTransformer encoder
                        cell. Default: "gelu".
            hidden_dropout_prob (float): The dropout probability for RobertaOutput. Default: 0.1.
            attention_probs_dropout_prob (float): The dropout probability for
                                          RobertaAttention. Default: 0.1.
            max_position_embeddings (int): Maximum length of sequences used in this
                                     model. Default: 512.
            type_vocab_size (int): Size of token type vocab. Default: 16.
            initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.

            use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
            dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
            compute_type (:class:`mindspore.dtype`): Compute type in RobertaTransformer. Default: mstype.float32.

            New parameters:

            bos_token_id (int): The id of the `beginning-of-stream` token.
            pad_token_id (int): The id of the `padding` token.
            eos_token_id (int): The id of the `end-of-stream` token.

            is_decoder(bool): Whether the model is used as decoder or not (in which case it's used as an encoder).
                            Default: False
            add_cross_attention (bool): Whether cross-attention layers should be added to the model. Note, this option
                                       is only relevant for models that can be used as decoder models within
                                    the `EncoderDecoderModel` class. Default: False
            tie_encoder_decoder (bool): Whether all encoder weights should be tied to their equivalent decoder weights.
                                    This requires the encoder and decoder model to have the exact same parameter names.
                                         Default: False
            output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return all hidden-states.

            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should returns all attentions.


        """

    def __init__(self,
                 config,
                 is_decoder=False,
                 add_cross_attention=False,

                 ):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.hidden_act = config.hidden_act
        self.intermediate_size = config.intermediate_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size
        self.initializer_range = config.initializer_range
        self.dtype = config.dtype
        self.compute_type = config.compute_type
        # roberta special
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        # as decoder
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.tie_encoder_decoder = config.tie_encoder_decoder

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.layer_norm_eps = config.layer_norm_eps

        self.use_cache = config.use_cache


class RobertaGenerationEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings.
       same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        # generation tasks don't need
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            tuple((config.hidden_size,)), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.zeros = ops.Zeros()
        self.dtype = config.dtype
        self.expand_dims = ops.ExpandDims()
        self.add = ops.Add()
        self.shape = ops.Shape()

    def construct(self, input_ids, past_key_values_length=0):
        """

        Args:
            input_ids:
            past_key_values_length:

        Returns:

        """
        position_ids = create_position_ids_from_input_ids(
            input_ids, self.padding_idx, past_key_values_length)

        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.add(embeddings, position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SaturateCast(nn.Cell):
    """
    Performs a safe saturating cast. This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        src_type (:class:`mindspore.dtype`): The type of the elements of the input tensor. Default: mstype.float32.
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor. Default: mstype.float32.
    """

    def __init__(self, src_type=mstype.float32, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        np_type = mstype.dtype_to_nptype(dst_type)

        self.tensor_min_type = float(np.finfo(np_type).min)
        self.tensor_max_type = float(np.finfo(np_type).max)

        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class RobertaSelfAttention(nn.Cell):
    """ self attention """

    def __init__(self, config: RobertaGenerationConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            logger.warning(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.attention_head_size_sqrt = math.sqrt(self.attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size).to_float(config.compute_type)
        self.key = nn.Dense(config.hidden_size, self.all_head_size).to_float(config.compute_type)
        self.value = nn.Dense(config.hidden_size, self.all_head_size).to_float(config.compute_type)
        self.dropout = nn.Dropout(
            p=config.attention_probs_dropout_prob).to_float(config.compute_type)
        self.is_decoder = config.is_decoder
        self.matmul = ops.BatchMatMul()
        self.Softmax = nn.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.print = ops.Print()
        self.concat_2 = ops.Concat(axis=2)
        self.compute_type = config.compute_type
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.cast = ops.Cast()

    def transpose_for_scores(self, x):
        """
        transpose the shape
        Args:
            x:

        Returns:

        """
        new_x_shape = x.shape[:-1] + \
                      (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        # x_shape (batch_size, num_attention_heads, seq_len, head_size)
        output = self.transpose(x, (0, 2, 1, 3))
        return output

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        """

        Args:
            hidden_states:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            past_key_value:
            output_attentions:

        Returns:

        """
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = self.concat_2([past_key_value[0], key_layer])
            value_layer = self.concat_2([past_key_value[1], value_layer])
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # mixed_key_layer = self.key(hidden_states)
            # mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.matmul(
            query_layer, self.transpose(key_layer, (0, 1, 3, 2)))

        attention_scores = attention_scores / self.attention_head_size_sqrt

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.Softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        context_layer = self.matmul(attention_probs, value_layer)
        # V reshape, [batch_size, length, embedding_dimension]
        context_layer = self.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs


class RobertaSelfOutput(nn.Cell):
    """self output  """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.add = ops.Add()

    def construct(self, hidden_states, input_tensor):
        """

        Args:
            hidden_states:
            input_tensor:

        Returns:

        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.add(hidden_states, input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RobertaAttention(nn.Cell):
    """ attention """

    def __init__(self, config):
        super().__init__()
        self.self_attention = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.print = ops.Print()

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        """

        Args:
            hidden_states:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            past_key_value:
            output_attentions:

        Returns:

        """
        self_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class RobertaIntermediate(nn.Cell):
    """ roberta  intermediate """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        """

        Args:
            hidden_states:

        Returns:

        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Cell):
    """ roberta output """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size,
                              config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.add = ops.Add()

    def construct(self, hidden_states, input_tensor):
        """

        Args:
            hidden_states:
            input_tensor:

        Returns:

        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.add(input_tensor, hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RobertaLayer(nn.Cell):
    """ roberta layer """

    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # use as decoder, add cross attention
        if self.is_decoder:
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        self.print = ops.Print()

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False
    ):
        """

        Args:
            hidden_states:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            past_key_value:
            output_attentions:

        Returns:

        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states=hidden_states,
                                                attention_mask=attention_mask,
                                                output_attentions=output_attentions,
                                                past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        outputs = None  # initial
        present_key_value = None
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # add self attentions if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions
            )
            attention_output = cross_attention_outputs[0]

            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs


class RobertaEncoderCell(nn.Cell):
    """ roberta encoder cell """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_cross_attention = config.add_cross_attention
        self.layer = nn.CellList([RobertaLayer(config)
                                  for _ in range(config.num_hidden_layers)])
        self.print = ops.Print()

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        """

        Args:
            hidden_states:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            past_key_values:
            use_cache:
            output_attentions:
            output_hidden_states:

        Returns:

        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                                           (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if next_decoder_cache is not None:
            outputs = outputs + (next_decoder_cache,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_self_attentions,)
        if all_cross_attentions is not None:
            outputs = outputs + (all_cross_attentions,)

        # last-layer hidden state,cache, (all hidden states), (all self attentions) ,(all cross attention)
        return outputs


class RobertaPooler(nn.Cell):
    """ poller layer """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        """

        Args:
            hidden_states:

        Returns:

        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaGenerationPreTrainedModel(nn.Cell):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    base_model_prefix = "roberta"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, RobertaGenerationConfig):
            logger.warning(
                "Parameter config in `{}(config)` should be an instance of class `RobertaGenerationConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = copy.deepcopy(config)
        self.initializer_range = config.initializer_range
        self.compute_type = config.compute_type

    def init_weights(self):
        """ Initialize the weights.
        """
        self.init_parameters_data()
        for _, cell in self.cells_and_names():

            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    TruncatedNormal(self.initializer_range), cell.weight.shape, cell.weight.dtype))
                cell.to_float(self.compute_type)

                # module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        0.0, cell.bias.shape, cell.bias.dtype))

            elif isinstance(cell, nn.Embedding):
                cell.embedding_table.set_data(init.initializer(TruncatedNormal(self.initializer_range),
                                                               cell.embedding_table.shape,
                                                               cell.embedding_table.dtype))
                if cell.padding_idx is not None:
                    cell.embedding_table.data[cell.padding_idx] = init.initializer('zeros',
                                                                                   cell.embedding_table.data[
                                                                                       cell.padding_idx].shape,
                                                                                   cell.embedding_table.data[
                                                                                       cell.padding_idx].dtype)
            elif isinstance(cell, nn.LayerNorm):
                cell.beta.set_data(init.initializer(
                    'zeros', cell.beta.shape, cell.beta.dtype))
                cell.to_float(self.compute_type)
                cell.gamma.set_data(init.initializer(
                    'ones', cell.gamma.shape, cell.gamma.dtype))

    def tie_weights(self):
        pass  # Overwrite for models


class RobertaGenerationEncoder(RobertaGenerationPreTrainedModel):
    """
           encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
           `optional`):
               Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
               the model is configured as a decoder.
           encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
               Mask to avoid performing attention on the padding token indices of the encoder input.
               This mask is used in the cross-attention if the model is configured as a decoder.
               Mask values selected in ``[0, 1]``:

               - 1 for tokens that are **not masked**,
               - 0 for tokens that are **masked**.

           labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
               Labels for computing the left-to-right language modeling loss (next word prediction).
               Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
               Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
               labels in ``[0, ..., config.vocab_size]``
           past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple
                having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks.
                Can be used to speed up decoding.

               If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
               (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
               instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
           use_cache (:obj:`bool`, `optional`):
               If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
               decoding (see :obj:`past_key_values`).

           Returns:
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        config = copy.deepcopy(config)
        self.config = config
        self.is_decoder = config.is_decoder
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.embeddings = RobertaGenerationEmbeddings(config)
        self.encoder = RobertaEncoderCell(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.zeros = ops.Zeros()
        self.dtype = config.dtype
        self.ones = ops.Ones()
        self.expand_dims = ops.ExpandDims()  # unsqueeze
        self.concat = ops.Concat(axis=-1)
        self.tile = ops.Tile()  # repeat
        self.broadcast_to = ops.BroadcastTo((self.num_hidden_layers, -1))
        self.shape = ops.Shape()
        self.cast = ops.Cast()
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def construct(
            self,
            input_ids,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        """
        Args:
            input_ids:
            attention_mask:
            encoder_hidden_states:
            encoder_attention_mask:
            past_key_values:
            use_cache:
            output_attentions:
            output_hidden_states:
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        if self.is_decoder:
            use_cache = use_cache if use_cache is not None else self.use_cache
        else:
            use_cache = False

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = self.ones(
                (batch_size, seq_length + past_key_values_length), self.dtype)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]

        encoder_extended_attention_mask = None  # initial
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = self.ones(
                    encoder_hidden_shape, self.dtype)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        # else:
        #     encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            self.cast_compute_type(embedding_output),
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int]):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = None
        if attention_mask.ndim == 3:
            extended_attention_mask = self.expand_dims(attention_mask, 1)
            # extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads,
            # seq_length, seq_length]
            if self.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = generate_arange_tensor(seq_length)

                s1 = self.expand_dims(self.expand_dims(seq_ids, 0), 0)
                s2 = self.expand_dims(self.expand_dims(seq_ids, 0), 2)
                causal_mask = self.tile(s1, (batch_size, seq_length, 1)) <= s2

                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.astype(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = self.concat(
                        (
                            self.ones((batch_size, seq_length, prefix_seq_len), causal_mask.dtype),
                            causal_mask,
                        )
                    )
                causal_mask_expand = self.expand_dims(causal_mask, 1)
                attention_mask_expand = self.expand_dims(self.expand_dims(attention_mask, 1), 2)
                # attention_mask[:, None, None, :]
                extended_attention_mask = causal_mask_expand * attention_mask_expand
            else:
                extended_attention_mask = self.expand_dims(
                    self.expand_dims(attention_mask, 1), 2)
        # else:
        #     logger.warning("Wrong shape for input_ids (shape {input_shape})
        #     or attention_mask (shape {attention_mask.shape})")
        # raise ValueError(
        #     f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        # )
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.astype(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.
        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        encoder_extended_attention_mask = None
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = self.expand_dims(
                encoder_attention_mask, 1)
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = self.expand_dims(
                self.expand_dims(encoder_attention_mask, 1), 2)
            # encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.astype(
            dtype=self.dtype)
        # fp16 compatibility   #astype->to
        if self.dtype == mstype.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == mstype.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        # else:
        #     logger.warning("{self.dtype} not recognized. `dtype` should be set to either `torch.float32`
        #     or `torch.float16`")
        # raise ValueError(
        #     f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
        # )
        return encoder_extended_attention_mask

    def get_head_mask(
            self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head mask if needed.
        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`,
            `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(
                head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = self.expand_dims(head_mask, -1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = self.broadcast_to(head_mask)
            head_mask = self.expand_dims(self.expand_dims(
                self.expand_dims(head_mask, 1), -1), -1)
            # head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.ndim == 2:
            head_mask = self.expand_dims(self.expand_dims(
                self.expand_dims(head_mask, 1), -1), -1)
            # head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        # assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        # switch to float if need + fp16 compatibility
        head_mask = head_mask.astype(dtype=self.dtype)
        return head_mask


class RobertaLMHead(nn.Cell):
    """ lm head """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(
            tuple((config.hidden_size,)), epsilon=config.layer_norm_eps)
        self.decoder = nn.Dense(
            config.hidden_size, config.vocab_size)
        self.zeros = ops.Zeros()
        self.bias = Parameter(self.zeros(config.vocab_size, mstype.float32))
        self.decoder.bias = self.bias
        self.gelu = ops.GeLU()

    def construct(self, hidden_states):
        """

        Args:
            hidden_states:

        Returns:

        """
        x = self.dense(hidden_states)
        x = self.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        """ tie weights """
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RobertaGenerationDecoder(RobertaGenerationPreTrainedModel):
    """ roberta decoder """

    def __init__(self, config):
        super(RobertaGenerationDecoder, self).__init__(config)

        # if not config.is_decoder:
        #     logger.warning("If you want to use `RobertaGenerationDecoder` as a standalone, add `is_decoder=True.`")

        config = copy.deepcopy(config)
        self.config = config

        self.vocab_size = config.vocab_size

        self.roberta = RobertaGenerationEncoder(
            config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()
        self.softmax_cross_entropy_with_logits = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True)
        self.print = ops.Print()
        self.arg_max_with_value = ops.ArgMaxWithValue(axis=1, keep_dims=True)
        self.compute_type = config.compute_type
        self.cast = ops.Cast()
        self.cast_dtype = SaturateCast(dst_type=config.dtype)

    def get_output_embeddings(self):
        """ get output embedding """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """ set output embedding """
        self.lm_head.decoder = new_embeddings

    def construct(
            self,
            input_ids=None,
            encoder_hidden_states=None,
            labels=None,
            attention_mask=None,
            position_ids=None,

            encoder_attention_mask=None,

            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
         `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having
         4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
           import torch
           tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
            config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            config.is_decoder = True
            model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
            config=config)
            inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
            outputs = model(**inputs)
            prediction_logits = outputs.logits
        """
        if labels is not None:
            use_cache = False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None

        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            loss_fct = self.softmax_cross_entropy_with_logits
            logits = shifted_prediction_scores.view(-1, self.vocab_size)
            logits = self.cast_dtype(logits)
            lm_loss = loss_fct(logits, labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((lm_loss,) + output) if lm_loss is not None else output

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None):
        """ prepare inputs for generation """
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        """ reorder cache """
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                                     for past_state in layer_past),)
        return reordered_past


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    :param padding_idx:
    :param input_ids:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    # mask = input_ids.ne(padding_idx).int()
    not_equal = ops.NotEqual()
    cumsum = ops.CumSum()
    mask = not_equal(input_ids, padding_idx)
    mask = mask.astype(ms.int32)
    incremental_indices = (cumsum(mask, 1).astype(
        mask.dtype) + past_key_values_length) * mask
    return incremental_indices.astype(ms.int32) + padding_idx
