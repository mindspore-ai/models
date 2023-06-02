# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import ops
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window


class STFT(nn.Cell):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length=None, window='hann'):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        self.cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:self.cutoff, :]),
                                   np.imag(fourier_basis[:self.cutoff, :])])

        self.forward_basis = ms.Tensor(fourier_basis[:, None, :], ms.float32)

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = ms.Tensor(fft_window, ms.float32)
            self.forward_basis *= fft_window

    def construct(self, input_data):
        input_data = ops.expand_dims(input_data, 1)
        input_data = ops.Pad(((0, 0), (0, 0), (int(self.filter_length / 2), int(self.filter_length / 2))))(input_data)
        forward_transform = nn.Conv1d(1, self.cutoff * 2, self.win_length, stride=self.hop_length, pad_mode='valid',
                                      weight_init=self.forward_basis)(input_data)

        real_part = forward_transform[:, :self.cutoff, :]
        imag_part = forward_transform[:, self.cutoff:, :]

        magnitude = ops.sqrt(real_part**2 + imag_part**2)
        phase = ops.atan2(imag_part, real_part)

        return magnitude, phase


class MelSpectrogram(nn.Cell):
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)
        mel_basis = mel(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax, htk=True)
        self.mel_basis = ms.Tensor(mel_basis, ms.float32)
        self.min_bound = ms.Tensor(1e-5, ms.float32)

    def construct(self, y):
        magnitudes, _ = self.stft(y)
        mel_output = ops.matmul(self.mel_basis, magnitudes)
        mel_output = ops.clip_by_value(mel_output, clip_value_min=self.min_bound)
        mel_output = ops.log(mel_output)
        return mel_output
