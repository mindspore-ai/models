# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Transformer model."""
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr
from .parallel_transformer import Dropout, TransformerEncoder, TransformerDecoder, VocabEmbedding, \
    set_parallel_configure_for_layer, BertAttentionMaskWithoutLen
from .beam_search import BeamSearchDecoder, TileBeam

class TransformerConfig:
    """
    Configuration for `Transformer`.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 36560.
        hidden_size (int): Size of the layers. Default: 1024.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder/decoder
                           cell. Default: 6.
        num_attention_heads (int): Number of attention heads in the Transformer
                             encoder/decoder cell. Default: 16.
        intermediate_size (int): Size of intermediate layer in the Transformer
                           encoder/decoder cell. Default: 4096.
        hidden_act (str): Activation function used in the Transformer encoder/decoder
                    cell. Default: "relu".
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.3.
        attention_probs_dropout_prob (float): The dropout probability for
                                      MultiheadAttention. Default: 0.3.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        label_smoothing (float): label smoothing setting. Default: 0.1
        beam_width (int): beam width setting. Default: 4
        max_decode_length (int): max decode length in evaluation. Default: 80
        length_penalty_weight (float): normalize scores of translations according to their length. Default: 1.0
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size=100,
                 seq_length=128,
                 vocab_size=36560,
                 hidden_size=1024,
                 num_hidden_layers=6,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=128,
                 initializer_range=0.02,
                 label_smoothing=0.1,
                 beam_width=1,
                 max_decode_length=30,
                 length_penalty_weight=1.0,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.dtype = dtype
        self.compute_type = compute_type


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        depth (int): Hidden size.
        min_timescale (float): Default: 1.
        max_timescale (float): Default: 10000.

    Returns:
        Tensor of shape (length, depth)
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """

    def __init__(self,
                 embedding_size,
                 max_position_embeddings=128,
                 dropout_prob=0.1,
                 parallel_config=None):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=mstype.float32)
        self.multiply = P.Mul().shard(((parallel_config.dp, 1, 1), (1,)))
        self.add = P.Add().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))
        self.dropout = Dropout(1 - dropout_prob, dtype=mstype.float32).shard(((parallel_config.dp, 1, 1),))
        self.use_dropout = dropout_prob > 0
        self.expand_dims = P.ExpandDims().shard(((1, 1),))
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.position_embedding_table = Tensor(position_encoding(max_position_embeddings, embedding_size),
                                               mstype.float32)
        self.shape = P.Shape()
        self.embedding_size = embedding_size

    def construct(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        input_shape = self.shape(word_embeddings)
        input_len = input_shape[1]

        output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        # position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.slice(self.position_embedding_table, (0, 0), (input_len, self.embedding_size),
                                         (1, 1))
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(output, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """

    def __init__(self, src_type=mstype.float32, dst_type=mstype.float32):
        super(CastWrapper, self).__init__()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)


class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        batch_size (int): Batch size.
        seq_length (int): Length of input sequence.
        width (int): Hidden size.
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.
        dtype (:class:`mindspore.dtype`): Compute type to compute log_softmax. Default: mstype.float32.
    """

    def __init__(self,
                 width,
                 compute_type=mstype.float32,
                 dtype=mstype.float32,
                 parallel_config=None):
        super(PredLogProbs, self).__init__()
        self.width = width
        self.compute_type = compute_type
        self.dtype = dtype

        self.reshape = P.Reshape()
        self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.dp, 1), (1, 1)))
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.log_softmax.log_softmax.shard(((parallel_config.dp, 1),))
        self.cast = P.Cast()

    def construct(self,
                  input_tensor,
                  output_weights,
                  seq_length):
        """Get log probs."""
        shape_flat_sequence_tensor = (-1, self.width)

        input_tensor = self.reshape(input_tensor, shape_flat_sequence_tensor)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (:class:`TransformerConfig`): Configuration for Transformer.
    """

    def __init__(self):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.batch_matmul = P.BatchMatMul()

    def construct(self, input_mask):
        """Create attention mask according to input mask."""
        input_shape = self.shape(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)

        input_mask = self.cast(input_mask, mstype.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.batch_matmul(mask_left, mask_right)

        return attention_mask


class TransformerDecoderStep(nn.Cell):
    """
    Multi-layer transformer decoder step.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        max_decode_length (int): Max decode length.
        enc_seq_length (int): Length of source sentences.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.
        embedding_lookup (:class:`EmbeddingLookup`): Embedding lookup module.
        embedding_processor (:class:`EmbeddingPostprocessor`) Embedding postprocessor module.
        projection (:class:`PredLogProbs`): PredLogProbs module
    """

    def __init__(self,
                 parallel_config,
                 hidden_size,
                 max_decode_length,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.3,
                 seq_length=30,
                 hidden_dropout_prob=0.3,
                 hidden_act="relu",
                 compute_type=mstype.float32,
                 embedding_lookup=None,
                 embedding_processor=None,
                 projection=None,
                 lambda_func=set_parallel_configure_for_layer,
                 offset=0
                 ):
        super(TransformerDecoderStep, self).__init__(auto_prefix=False)
        self.num_hidden_layers = num_hidden_layers

        self.tfm_embedding_lookup = embedding_lookup
        self.tfm_embedding_processor = embedding_processor
        self.projection = projection

        self.tfm_decoder = TransformerDecoder(num_layers=self.num_hidden_layers,
                                              hidden_size=hidden_size,
                                              ffn_hidden_size=intermediate_size,
                                              num_heads=num_attention_heads,
                                              seq_length=seq_length,
                                              attention_dropout_rate=attention_probs_dropout_prob,
                                              hidden_dropout_rate=hidden_dropout_prob,
                                              hidden_act=hidden_act,
                                              post_layernorm_residual=False,
                                              use_moe=False,
                                              parallel_config=parallel_config,
                                              lambda_func=lambda_func,
                                              offset=(1 + offset) * parallel_config.fusion_group)

        self.ones_like = P.OnesLike()
        self.shape = P.Shape()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()
        self.tile = P.Tile()
        ones = np.ones(shape=(max_decode_length, max_decode_length))
        self.future_mask = Tensor(np.tril(ones), dtype=mstype.float32)

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.beam_width = 1
        self.tile_beam = TileBeam(beam_width=self.beam_width)

    # def construct(self, input_ids, enc_states, source_mask, seq_length):
    def construct(self, input_ids, enc_states, source_mask):
        """
        Multi-layer transformer decoder step.
        input_ids: [batch_size * beam_width]
        """
        # process embedding
        input_embedding, embedding_tables = self.tfm_embedding_lookup(input_ids)
        input_embedding = self.tfm_embedding_processor(input_embedding)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]
        future_mask = self.future_mask[0:input_len:1, 0:input_len:1]

        input_mask = self.ones_like(input_ids)
        input_mask = self._create_attention_mask_from_input_mask(input_mask)
        input_mask = self.multiply(input_mask, self.expand(future_mask, 0))
        input_mask = self.expand(input_mask, 1)
        # input_mask = self.tile(input_mask, (1, input_ids.shape[1], 1, 1))
        input_mask = self.cast_compute_type(input_mask)
        # print(enc_attention_mask.shape)
        # enc_attention_mask = enc_attention_mask[::, 0:1:1, 0:input_len:1]
        # enc_attention_mask = source_mask.view(source_mask.shape[0], 1, 1, seq_length)
        enc_attention_mask = source_mask.view(source_mask.shape[0], 1, 1, source_mask.shape[1])
        enc_attention_mask = self.tile(enc_attention_mask, (1, 1, input_len, 1))

        beam_enc_attention_mask = self.tile_beam(enc_attention_mask)
        beam_enc_attention_mask = self.cast_compute_type(beam_enc_attention_mask)

        # call TransformerDecoder
        # print(input_embedding.shape, input_mask.shape, enc_states.shape, enc_attention_mask.shape)
        decoder_output, _, _ = self.tfm_decoder(input_embedding, input_mask, enc_states, beam_enc_attention_mask)

        # take the last step
        decoder_output = decoder_output[::, input_len - 1:input_len:1, ::]

        # projection and log_prob
        log_probs = self.projection(decoder_output, embedding_tables, 1)

        return log_probs


@constexpr
def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    tril_ones = Tensor(np.tril(ones).reshape(1, 1, seq_length, seq_length), dtype=mstype.float32)
    return tril_ones


class TransformerModel(nn.Cell):
    """
    Transformer with encoder and decoder.

    Args:
        config (Class): Configuration for Transformer.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """

    def __init__(self,
                 config,
                 is_training,
                 input_size,
                 parallel_config,
                 offset=0,
                 use_moe=False):
        super(TransformerModel, self).__init__()
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        # self.attention_mask_enc = AttentionMask(30, parallel_config)
        # self.attention_mask_dec = AttentionMask(config.seq_length, parallel_config)

        self.attention_mask_enc = BertAttentionMaskWithoutLen(parallel_config)
        self.attention_mask_dec = BertAttentionMaskWithoutLen(parallel_config)

        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size

        self.last_idx = self.num_hidden_layers - 1
        self.beam_width = config.beam_width
        self.max_decode_length = config.max_decode_length

        self.tfm_embedding_lookup = VocabEmbedding(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            parallel_config=parallel_config)
        self.tfm_embedding_postprocessor_for_encoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            parallel_config=parallel_config)
        self.tfm_embedding_postprocessor_for_decoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            parallel_config=parallel_config)

        lambda_func = None if not parallel_config.optimizer_shard else set_parallel_configure_for_layer
        self.tfm_encoder = TransformerEncoder(num_layers=self.num_hidden_layers,
                                              hidden_size=self.hidden_size,
                                              ffn_hidden_size=config.intermediate_size,
                                              num_heads=config.num_attention_heads,
                                              seq_length=config.seq_length,
                                              attention_dropout_rate=config.attention_probs_dropout_prob,
                                              hidden_dropout_rate=config.hidden_dropout_prob,
                                              hidden_act=config.hidden_act,
                                              post_layernorm_residual=False,
                                              use_moe=use_moe,
                                              parallel_config=parallel_config,
                                              lambda_func=lambda_func,
                                              offset=offset * parallel_config.fusion_group)
        if is_training:
            self.projection = PredLogProbs(
                width=self.hidden_size,
                compute_type=config.compute_type,
                dtype=config.dtype,
                parallel_config=parallel_config)

            self.tfm_decoder = TransformerDecoder(num_layers=self.num_hidden_layers,
                                                  hidden_size=self.hidden_size,
                                                  ffn_hidden_size=config.intermediate_size,
                                                  num_heads=config.num_attention_heads,
                                                  seq_length=config.seq_length,
                                                  attention_dropout_rate=config.attention_probs_dropout_prob,
                                                  hidden_dropout_rate=config.hidden_dropout_prob,
                                                  hidden_act=config.hidden_act,
                                                  post_layernorm_residual=False,
                                                  use_moe=False,
                                                  parallel_config=parallel_config,
                                                  lambda_func=lambda_func,
                                                  offset=(1 + offset) * parallel_config.fusion_group)
        else:
            self.projection = PredLogProbs(
                width=self.hidden_size,
                compute_type=config.compute_type,
                dtype=config.dtype,
                parallel_config=parallel_config)
            self.tfm_decoder = TransformerDecoderStep(
                parallel_config=parallel_config,
                hidden_size=self.hidden_size,
                max_decode_length=config.max_decode_length,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                seq_length=config.seq_length,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_act=config.hidden_act,
                compute_type=config.compute_type,
                embedding_lookup=self.tfm_embedding_lookup,
                embedding_processor=self.tfm_embedding_postprocessor_for_decoder,
                projection=self.projection,
                lambda_func=lambda_func,
                offset=(1 + offset) * parallel_config.fusion_group
            )
            self.tfm_decoder = BeamSearchDecoder(
                batch_size=self.batch_size,
                seq_length=config.seq_length,
                vocab_size=config.vocab_size,
                decoder=self.tfm_decoder,
                beam_width=config.beam_width,
                length_penalty_weight=config.length_penalty_weight,
                max_decode_length=config.max_decode_length,
                sos_id=0, eos_id=0)

            self.tfm_decoder.add_flags(loop_can_unroll=True)
            self.tile_beam = TileBeam(beam_width=self.beam_width)
            ones = np.ones(shape=(self.batch_size, self.max_decode_length))
            self.encdec_mask = Tensor(ones, mstype.float32)
        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = CastWrapper(dst_type=config.compute_type)
        self.expand = P.ExpandDims()
        self.multiply = P.Mul().shard(((parallel_config.dp, 1, 1, 1), (1, 1, 1, 1)))
        self.shape = P.Shape()
        self.LinearEmbedding = nn.Dense(input_size, config.hidden_size).to_float(mstype.float16)
        self.LinearEmbedding.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.LinearEmbedding.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.LinearEmbedding.weight.parallel_optimizer = False
        self.LinearEmbedding.bias.parallel_optimizer = False
        self.tile = P.Tile().shard(((parallel_config.dp, 1, 1, 1),))
        self.zeros = P.Zeros()
        self.concat = P.Concat(axis=1)
        # self.tile_beam = TileBeam(beam_width=self.beam_width)

    def construct(self, sequence_output, source_mask, target_ids=None, target_mask=None):
        """Transformer with encoder and decoder."""
        src_embedding_output = self.LinearEmbedding(sequence_output.view(-1, sequence_output.shape[-1]))
        src_embedding_output = src_embedding_output.view(sequence_output.shape[0], sequence_output.shape[1], -1)

        # attention mask [batch_size, seq_length, seq_length]
        source_mask = self.cast_compute_type(source_mask)
        enc_attention_mask = self.attention_mask_enc(source_mask)
        # transformer encoder
        encoder_output, _, _ = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                                self.cast_compute_type(enc_attention_mask))

        if not self.is_training:
            #ipdb.set_trace()
            beam_encoder_output = self.tile_beam(encoder_output)
            seq_length = 30
            # enc_attention_mask = self.multiply(enc_attention_mask[::, 0:1:1, ::], self.expand(self.encdec_mask, -1))

            predicted_ids = self.tfm_decoder(beam_encoder_output, source_mask)
            ret = predicted_ids
        else:
            bos_sequence = self.zeros((target_ids.shape[0]), target_ids.dtype)
            bos_sequence = self.expand(bos_sequence, 1)
            target_ids = self.concat((bos_sequence, target_ids[:, :-1]))

            seq_length = target_ids.shape[1]
            future_mask = convert_np_to_tensor_encoder(seq_length)
            # process target sentence
            tgt_word_embeddings, embedding_tables = self.tfm_embedding_lookup(target_ids)
            tgt_embedding_output = self.tfm_embedding_postprocessor_for_decoder(tgt_word_embeddings)
            # attention mask [batch_size, seq_length, seq_length]
            tgt_attention_mask = self.attention_mask_dec(target_mask)
            tgt_attention_mask = self.multiply(tgt_attention_mask, future_mask)
            # transformer decoder
            enc_attention_mask = source_mask.view(source_mask.shape[0], 1, 1, source_mask.shape[1])
            enc_attention_mask = self.tile(enc_attention_mask, (1, 1, seq_length, 1))
            decoder_output, _, _ = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                                    self.cast_compute_type(tgt_attention_mask),
                                                    encoder_output, enc_attention_mask)
            # calculate logits and log_probs
            log_probs = self.projection(decoder_output, embedding_tables, seq_length)
            ret = log_probs

        return ret
