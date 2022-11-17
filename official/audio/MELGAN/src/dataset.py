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
"""dataset process"""
import glob
import random
import numpy as np
from scipy.io.wavfile import read

def read_wav_np(path):
    """read_wav_np"""
    sr, wav = read(path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    return sr, wav


class Generator1D:
    """Generate 1D"""
    def __init__(self, data_path, train_length, hop_size):
        self.wav_list = glob.glob(data_path + '/**/*.wav', recursive=True)
        random.shuffle(self.wav_list)

        self.mel_segment_length = train_length
        self.wav_segment_length = train_length * hop_size
        self.hop_size = hop_size

    def __getitem__(self, index):
        _, audio_y = read_wav_np(self.wav_list[index])
        mel_file = self.wav_list[index].replace('wav', 'mel')
        mel_file = mel_file.replace('.mel', '.npy')
        mel_y = np.load(mel_file)

        max_mel_start = mel_y.shape[1] - self.mel_segment_length - 1

        mel_start = random.randint(0, max_mel_start)
        mel_end = mel_start + self.mel_segment_length
        mel = mel_y[:, mel_start:mel_end]
        audio_start = mel_start * self.hop_size
        audio = audio_y[audio_start:audio_start + self.wav_segment_length]

        # for Discriminator
        mel_y = np.load(mel_file)

        max_meld_start = mel_y.shape[1] - self.mel_segment_length - 1
        meld_start = random.randint(0, max_meld_start)
        meld_end = meld_start + self.mel_segment_length
        meld = mel_y[:, meld_start:meld_end]
        audiod_start = meld_start * self.hop_size
        audiod = audio_y[audiod_start:audiod_start + self.wav_segment_length]

        batch_data = mel
        batch_wav = audio[np.newaxis, :]
        batch_datad = meld
        batch_wavd = audiod[np.newaxis, :]
        return batch_data, batch_wav, batch_datad, batch_wavd

    def __len__(self):
        return len(self.wav_list)
