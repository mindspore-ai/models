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
"""Audio parser script."""
import librosa
import numpy as np
import soundfile as sf


class LoadAudioAndTranscript:
    """
    Parse audio and transcript.
    """
    def __init__(
            self,
            audio_conf=None,
            normalize=False,
            labels=None
    ):
        super().__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sampling_rate']
        self.window = audio_conf['window']
        self.is_normalization = normalize
        self.labels = labels

    def load_audio(self, path):
        """
        Load audio.
        """
        sound, _ = sf.read(path, dtype='int16')
        sound = sound.astype('float32') / 32767
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)

        return sound

    def parse_audio(self, audio_path):
        """
        Parse audio.
        """
        audio = self.load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        d = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=self.window)
        mag, _ = librosa.magphase(d)
        mag = np.log1p(mag)
        if self.is_normalization:
            mean = mag.mean()
            std = mag.std()
            mag = (mag - mean) / std

        return mag
