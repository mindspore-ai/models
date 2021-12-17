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
""" FastSpeech2 Loss """
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms


class FastSpeech2ThreeV3Loss(nn.Cell):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2ThreeV3Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        self.log = ops.Log()
        self.unsqueeze = ops.ExpandDims()
        self.masked_select = ops.MaskedSelect()

    def construct(self, mel_targets, src_masks, mel_masks, duration_targets, pitch_targets, energy_targets,
                  mel_predictions, postnet_mel_predictions,
                  log_duration_predictions, pitch_predictions, energy_predictions):
        """Fastspeech2ThreeV3Loss construct"""
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = self.log(duration_targets.astype(ms.float32) + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        pitch_targets = pitch_targets.astype(ms.float32)
        energy_targets = energy_targets.astype(ms.float32)

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = self.masked_select(pitch_predictions, src_masks)
            pitch_targets = self.masked_select(pitch_targets, src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = self.masked_select(pitch_predictions, mel_masks)
            pitch_targets = self.masked_select(pitch_targets, mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = self.masked_select(energy_predictions, src_masks)
            energy_targets = self.masked_select(energy_targets, src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = self.masked_select(energy_predictions, mel_masks)
            energy_targets = self.masked_select(energy_targets, mel_masks)

        log_duration_predictions = self.masked_select(log_duration_predictions, src_masks)
        log_duration_targets = self.masked_select(log_duration_targets, src_masks)

        mel_predictions = self.masked_select(mel_predictions, self.unsqueeze(mel_masks, -1))
        postnet_mel_predictions = self.masked_select(postnet_mel_predictions, self.unsqueeze(mel_masks, -1))
        mel_targets = self.masked_select(mel_targets, self.unsqueeze(mel_masks, -1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return total_loss, mel_loss, postnet_mel_loss, duration_loss, pitch_loss, energy_loss
