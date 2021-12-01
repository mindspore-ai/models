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
fastspeech2
"""
import os
import json

import mindspore as ms
from mindspore import nn, ops

from fastspeech2_ms.transformer import Encoder, Decoder, PostNet
from fastspeech2_ms.utils.tools import get_mask_from_lengths
from .modules import VarianceAdaptor


class FastSpeech2ThreeV3(nn.Cell):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, input_size):
        super(FastSpeech2ThreeV3, self).__init__()
        self.model_config = model_config
        self.pre_encoder = nn.Dense(input_size, model_config["transformer"]["encoder_hidden"])
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Dense(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        self.cast = ops.Cast()
        self.unsqueezee = ops.ExpandDims()

    def construct(
            self,
            input_data,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):
        """FastSpeech2ThreeV3 construct"""
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        input_data = self.cast(input_data, ms.float32)

        input_data = self.pre_encoder(input_data)  # 48, 45, 256
        input_data = input_data.mean(1)  # 48, 256

        output = self.encoder(texts, src_masks)

        output[:, 0, :] = output[:, 0, :] + input_data

        if self.speaker_emb is not None:
            speaker_emb = self.speaker_emb(speakers)
            batch_size, emb_size = speaker_emb.shape[0], speaker_emb.shape[1]
            speaker_emb = self.unsqueezee(speaker_emb, 1)
            broadcast_to = ops.BroadcastTo((batch_size, max_src_len.asnumpy().tolist(), emb_size))
            output = output + broadcast_to(speaker_emb)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
