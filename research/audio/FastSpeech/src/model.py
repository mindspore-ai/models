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
"""FastSpeech model."""
import mindspore.numpy as msnp
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import XavierUniform
from mindspore.common.initializer import initializer

from src.cfg.config import config as hp
from src.modules import CBHG
from src.modules import LengthRegulator
from src.transformer.models import Decoder
from src.transformer.models import Encoder


class FastSpeech(nn.Cell):
    """FastSpeech model."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            n_src_vocab=hp.vocab_size,
            len_max_seq=hp.vocab_size,
            d_word_vec=hp.encoder_dim,
            n_layers=hp.encoder_n_layer,
            n_head=hp.encoder_head,
            d_k=hp.encoder_dim // hp.encoder_head,
            d_v=hp.encoder_dim // hp.encoder_head,
            d_model=hp.encoder_dim,
            d_inner=hp.encoder_conv1d_filter_size,
            dropout=hp.dropout,
        )

        self.length_regulator = LengthRegulator()

        self.decoder = Decoder(
            len_max_seq=hp.max_seq_len,
            n_layers=hp.decoder_n_layer,
            n_head=hp.decoder_head,
            d_k=hp.decoder_dim // hp.decoder_head,
            d_v=hp.decoder_dim // hp.decoder_head,
            d_model=hp.decoder_dim,
            d_inner=hp.decoder_conv1d_filter_size,
            dropout=hp.dropout
        )

        num_mels = hp.num_mels
        decoder_dim = hp.decoder_dim

        self.mel_linear = nn.Dense(
            decoder_dim,
            num_mels,
            weight_init=initializer(
                XavierUniform(),
                [num_mels, decoder_dim],
                mstype.float32
            )
        )

        self.last_linear = nn.Dense(
            num_mels * 2,
            num_mels,
            weight_init=initializer(
                XavierUniform(),
                [num_mels, num_mels * 2],
                mstype.float32
            )
        )

        self.postnet = CBHG(
            in_dim=num_mels,
            num_banks=8,
            projections=[256, hp.num_mels],
        )

        self.expand_dims = ops.ExpandDims()
        self.argmax = ops.ArgMaxWithValue(axis=-1)
        self.broadcast = ops.BroadcastTo((-1, -1, num_mels))

        self.ids_linspace = msnp.arange(hp.mel_max_length)
        self.zeros_mask = msnp.zeros((hp.batch_size, hp.mel_max_length, hp.num_mels))

    def mask_tensor(self, mel_output, position):
        """
        Make mask for tensor, to ignore padded cells.
        """
        lengths = self.argmax(position)[1]

        ids = self.ids_linspace

        mask = (ids < self.expand_dims(lengths, 1)).astype(mstype.float32)
        mask_bool = self.broadcast(self.expand_dims(mask, -1)).astype(mstype.bool_)

        mel_output = msnp.where(mask_bool, mel_output, self.zeros_mask)

        return mel_output

    def construct(
            self,
            src_seq,
            src_pos,
            mel_pos=None,
            mel_max_length=None,
            length_target=None,
            alpha=1.0
    ):
        """
        Predict mel-spectrogram from sequence.

        Args:
            src_seq (Tensor): Tokenized text sequence. Shape (hp.batch_size, hp.character_max_length)
            src_pos (Tensor): Positions of the sequences. Shape (hp.batch_size, hp.character_max_length)
            mel_pos (Tensor): Positions of the mels. Shape (hp.batch_size, hp.mel_max_length)
            mel_max_length (int): Max mel length.
            length_target (Tensor): Duration of the each phonema. Shape (hp.batch_size, hp.character_max_length)
            alpha (int): Regulator of the speech speed.
        """
        encoder_output = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(
                encoder_output,
                target=length_target,
                alpha=alpha,
                mel_max_length=mel_max_length,
            )

            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos)

            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)

            mel_postnet_output = mel_output + residual
            mel_postnet_output = self.mask_tensor(mel_postnet_output, mel_pos)

            return mel_output, mel_postnet_output, duration_predictor_output

        length_regulator_output, decoder_pos, mel_len = self.length_regulator(encoder_output, alpha=alpha)

        decoder_output = self.decoder(length_regulator_output, decoder_pos)

        mel_output = self.mel_linear(decoder_output)

        residual = self.postnet(mel_output)
        residual = self.last_linear(residual)

        mel_postnet_output = mel_output + residual

        return mel_output, mel_postnet_output, mel_len


class LossWrapper(nn.Cell):
    """
    Training wrapper for model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def construct(
            self,
            character,
            src_pos,
            mel_pos,
            duration,
            mel_target,
            max_mel_len,
    ):
        """
        FastSpeech with loss.

        Args:
            character (Tensor): Tokenized text sequence. Shape (hp.batch_size, hp.character_max_length)
            src_pos (Tensor): Positions of the sequences. Shape (hp.batch_size, hp.character_max_length)
            mel_pos (Tensor): Positions of the mels. Shape (hp.batch_size, hp.mel_max_length)
            duration (Tensor): Target duration. Shape (hp.batch_size, hp.character_max_length)
            mel_target (Tensor): Target mel-spectrogram. Shape (hp.batch_size, hp.mel_max_length, hp.num_mels)
            max_mel_len (list): Max mel length.

        Returns:
            total_loss (Tensor): Sum of 3 losses.
        """
        max_mel_len = max_mel_len[0]
        mel_output, mel_postnet_output, duration_predictor_output = self.model(
            character,
            src_pos,
            mel_pos=mel_pos,
            mel_max_length=max_mel_len,
            length_target=duration,
        )

        mel_loss = self.mse_loss(mel_output, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet_output, mel_target)
        duration_predictor_loss = self.l1_loss(duration_predictor_output, duration)

        total_loss = mel_loss + mel_postnet_loss + duration_predictor_loss

        return total_loss


class FastSpeechEval:
    """FastSpeech with vocoder for evaluation."""
    def __init__(
            self,
            mel_generator,
            vocoder,
            config,
    ):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder

        self.alpha = config.alpha
        self.vocoder_stride = vocoder.upsample.stride[1]
        self.zeros_mask = msnp.zeros((1, config.num_mels, config.mel_max_length))

        x_grid = msnp.arange(0, config.mel_max_length)
        y_grid = msnp.arange(0, config.num_mels)

        self.transpose = ops.Transpose()
        self.grid = ops.ExpandDims()(msnp.meshgrid(x_grid, y_grid)[0], 0)

    def get_audio(self, src_seq, src_pos):
        """
        Generate mel-spectrogram from sequence,
        generate raw audio from mel-spectrogram by vocoder.
        """
        _, mel, mel_len = self.mel_generator(src_seq, src_pos, alpha=self.alpha)

        mel_mask = (self.grid < mel_len).astype(mstype.float32)
        clear_mel = self.transpose(mel, (0, 2, 1)) * mel_mask

        audio = self.vocoder.construct(clear_mel)

        audio_len = mel_len * self.vocoder_stride

        return audio, audio_len
