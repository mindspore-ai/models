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
"""inference model"""


import mindspore.nn as nn
from src.roberta_model import RobertaGenerationEncoder, RobertaGenerationDecoder
from src.beam_search import BeamSearchDecoder, TileBeam
from src.model_encoder_decoder import EncoderDecoderConfig


class EncoderDecoderInferModel(nn.Cell):
    """
    Args:
        config:
        encoder:
        decoder:
        is_training:
        add_pooling_layer:
    """
    def __init__(self, config=None, encoder=None, decoder=None, is_training=True, add_pooling_layer=True):
        assert config is not None or (encoder is not None and decoder is not None
                                      ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig(
                encoder=encoder.config, decoder=decoder.config)

        super().__init__(config)

        if encoder is None:
            encoder = RobertaGenerationEncoder(config.encoder, add_pooling_layer=add_pooling_layer)

        if decoder is None:
            decoder = RobertaGenerationDecoder(config.decoder)

        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.is_training = is_training

        self.num_hidden_layers = config.num_hidden_layers
        self.beam_width = config.beam_width
        self.max_decode_length = config.max_decode_length

        self.max_length = config.max_length
        self.length_penalty_weight = config.length_penalty_weight

        self.batch_size = config.batch_size

        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.compute_type = config.compute_type

        self.beam_decoder = BeamSearchDecoder(
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
            decoder=self.decoder,
            beam_width=config.beam_width,
            length_penalty_weight=config.length_penalty_weight,
            max_decode_length=config.max_decode_length,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            compute_type=self.compute_type
        )
        self.tile_beam = TileBeam(
            beam_width=self.beam_width, compute_type=self.compute_type)

        self.log_softmax = nn.LogSoftmax(axis=-1)

    def construct(self,
                  input_ids=None,
                  attention_mask=None,
                  output_attentions=None,
                  output_hidden_states=None,

                  ):
        """

        Args:
            input_ids:
            attention_mask:
            output_attentions:
            output_hidden_states:

        Returns:

        """
        # encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_hidden_states = encoder_outputs[0]
        beam_encoder_output = self.tile_beam(encoder_hidden_states)
        beam_enc_attention_mask = self.tile_beam(attention_mask)
        predicted_ids = self.beam_decoder(
            beam_encoder_output, beam_enc_attention_mask)
        return predicted_ids
