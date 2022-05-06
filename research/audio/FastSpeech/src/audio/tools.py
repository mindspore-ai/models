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
"""Preprocessing tools."""
import numpy as np
from scipy.io.wavfile import read

from src.audio import stft
from src.cfg.config import config

_stft = stft.TacotronSTFT(
    config.au_filter_length,
    config.au_hop_length,
    config.au_win_length,
    config.au_n_mel_channels,
    config.au_sampling_rate,
    config.au_mel_fmin,
    config.au_mel_fmax,
)


def load_wav_to_array(full_path):
    """Load wav file as numpy array."""
    sampling_rate, data = read(full_path)
    return data.astype(np.float32), sampling_rate


def get_mel(filename):
    """Process loaded audio to mel-spectrogram."""
    audio, _ = load_wav_to_array(filename)
    audio_norm = audio / config.au_max_wav_value
    audio_norm = np.expand_dims(audio_norm, 0)
    melspec = _stft.mel_spectrogram(audio_norm)
    melspec = np.squeeze(melspec, 0)

    return melspec
