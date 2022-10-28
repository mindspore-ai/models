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
"""EncoderDecoder model"""

from mindspore import log as logger
import mindspore.nn as nn
from src.roberta_model import RobertaGenerationEncoder, RobertaGenerationDecoder


class EncoderDecoderConfig:
    '''
    Args:
        name_or_path (str): Name or path to the pretrained checkpoint
        output_hidden_states (bool): Whether or not the model should return all hidden-states. Default: False
        output_attentions (bool): Whether or not the model should returns all attentions. Default: False
        tie_word_embedding (bool): Whether input and output word embeddings should be tied for Seq2Seq models.
            Default: True
        add_cross_attention (bool): Whether cross-attention layers should be added to the model. Default: False
        tie_encoder_decoder (bool): Whether all encoder weights should be tied to their equivalent decoder weights.
            This requires the encoder and decoder model to have the exact same parameter names. Default: False
        chunk_size_feed_forward (int):  The chunk size of all feed forward layers in the residual attention blocks.
            A chunk size of 0 means that the feed forward layer is not chunked. Default: 0

    Parameters for sequence generation:
        max_length (int): Maximum length that will be used by default in the generate method of the model. Default: 20
        min_length (int): Minimum length that will be used by default in the generate method of the model. Default: 10
        do_sample (bool): Flag that will be used by default in the generate method of the model.
            Whether or not to use sampling ; use greedy decoding otherwise. Default: False
        early_stopping (bool): Whether to stop the beam search when at least ``num_beams``
          sentences are finished per batch or not. Default: False
        num_beams (int): Number of beams for beam search that will be used by
          default in the :obj:`generate` method of the model. 1 means no beam search. Default: 1
        num_beam_groups (int): Number of groups to divide :obj:`num_beams`
          into in order to ensure diversity among different groups of beams that will be used by default in the
          :obj:`generate` method of the model. 1 means no group beam search. Default:1

    '''

    def __init__(self, encoder, decoder, max_length=20, beam_width=1, max_decode_length=64, length_penalty_weight=1.0,
                 batch_size=1, tie_encoder_decoder=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_encoder_decoder = True
        self.tie_encoder_decoder = tie_encoder_decoder
        self.max_length = max_length
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.batch_size = batch_size

        self.num_hidden_layers = decoder.num_hidden_layers
        self.vocab_size = decoder.vocab_size
        self.pad_token_id = self.encoder.pad_token_id
        self.bos_token_id = self.encoder.bos_token_id
        self.eos_token_id = self.encoder.eos_token_id
        self.compute_type = self.encoder.compute_type


class EncoderDecoderModel(nn.Cell):
    """
    Args:
        config:
        encoder:
        decoder:
        is_training:
        tie_encoder_decoder:
    """

    def __init__(self, config=None, encoder=None, decoder=None, is_training=True, tie_encoder_decoder=True):
        assert config is not None or (encoder is not None and decoder is not None), \
            "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig(
                encoder=encoder.config, decoder=decoder.config, tie_encoder_decoder=tie_encoder_decoder)

        super().__init__(config)

        if encoder is None:
            encoder = RobertaGenerationEncoder(config.encoder)

        if decoder is None:
            decoder = RobertaGenerationDecoder(config.decoder)

        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.is_training = is_training
        self.log_softmax = nn.LogSoftmax(axis=-1)
        # tie encoder, decoder weights if config set accordingly
        if config.tie_encoder_decoder:
            self.tie_weights()

    def _tie_encoder_to_decoder_recursively(
            self,
            decoder_pointer,
            encoder_pointer,
            module_name,
            uninitialized_encoder_weights,
            depth=0,
    ):
        """

        Args:
            decoder_pointer:
            encoder_pointer:
            module_name:
            uninitialized_encoder_weights:
            depth:

        Returns:

        """
        assert isinstance(decoder_pointer, nn.Cell) and isinstance(
            encoder_pointer, nn.Cell
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Cell"
        if hasattr(decoder_pointer, "weight"):
            assert hasattr(encoder_pointer, "weight")
            # encoder_pointer.weight.set_data(decoder_pointer.weight.data)
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                # encoder_pointer.bias.set_data(decoder_pointer.bias.data)
                encoder_pointer.bias = decoder_pointer.bias
            return
        if hasattr(decoder_pointer, 'gamma'):
            assert hasattr(encoder_pointer, 'gamma')
            encoder_pointer.gamma.set_data(decoder_pointer.gamma.data)
            encoder_pointer.gamma = decoder_pointer.gamma
            if hasattr(decoder_pointer, 'beta'):
                assert hasattr(encoder_pointer, 'beta')
                # encoder_pointer.beta.set_data(decoder_pointer.beta.data)
                encoder_pointer.beta = decoder_pointer.beta
            return

        if hasattr(decoder_pointer, "embedding_table"):
            assert hasattr(encoder_pointer, "embedding_table")
            # encoder_pointer.embedding_table.set_data(decoder_pointer.embedding_table.data)
            encoder_pointer.embedding_table = decoder_pointer.embedding_table
            return
        encoder_cells = encoder_pointer.cells()
        decoder_cells = decoder_pointer.cells()
        if decoder_cells:
            assert (
                encoder_cells
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_cells.keys()}
            encoder_layer_pos = 0
            for name, _ in decoder_cells.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_cells[decoder_name], type(encoder_cells[encoder_name])) and len(
                            encoder_cells
                    ) != len(decoder_cells):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_cells:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is "
                        "a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                self._tie_encoder_to_decoder_recursively(
                    decoder_cells[decoder_name],
                    encoder_cells[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(
                    module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # base_model_prefix :str
    # @staticmethod
    def _tie_encoder_decoder_weights(self, encoder, decoder, base_model_prefix):
        """

        Args:
            encoder:
            decoder:
            base_model_prefix:

        Returns:

        """
        uninitialized_encoder_weights = []
        # tie weights recursively
        self._tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
        if uninitialized_encoder_weights:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder.cells()[decoder_base_model_prefix], self.decoder.base_model_prefix)

    def construct(self,
                  input_ids=None,
                  decoder_input_ids=None,
                  labels=None,
                  attention_mask=None,
                  decoder_attention_mask=None,
                  encoder_outputs=None,
                  past_key_values=None,
                  use_cache=None,
                  output_attentions=None,
                  output_hidden_states=None,
                  ):
        """

        Args:
            input_ids:
            decoder_input_ids:
            labels:
            attention_mask:
            decoder_attention_mask:
            encoder_outputs:
            past_key_values:
            use_cache:
            output_attentions:
            output_hidden_states:

        Returns:

        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        encoder_hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        return decoder_outputs + encoder_outputs
