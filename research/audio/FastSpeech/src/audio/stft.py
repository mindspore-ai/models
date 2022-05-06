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
"""Tacotron module."""
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import Conv2D
from scipy.signal import get_window


class STFT:
    """Mel-spectrogram transformer."""
    def __init__(
            self,
            filter_length=800,
            hop_length=200,
            win_length=800,
            window='hann'
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [
                np.real(fourier_basis[:cutoff, :]),
                np.imag(fourier_basis[:cutoff, :])
            ]
        )

        forward_basis = fourier_basis[:, None, :].astype(np.float32)
        inverse_basis = np.linalg.pinv(scale * fourier_basis).T[:, None, :].astype(np.float32)

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = np.array(fft_window, np.float32)

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.forward_basis = forward_basis.astype(np.float32)
        self.inverse_basis = inverse_basis.astype(np.float32)

        self.conv = Conv2D(
            out_channel=self.forward_basis.shape[0],
            kernel_size=self.forward_basis.shape[1:],
            stride=self.hop_length,
            pad_mode='pad',
            pad=0
        )

    def transform(self, input_data):
        """Transforms input wav to raw mel-spect data."""
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]

        input_data = input_data.reshape(num_batches, 1, num_samples)
        input_data = np.pad(np.squeeze(input_data), int(self.filter_length / 2), mode='reflect')

        input_data = np.expand_dims(np.expand_dims(np.expand_dims(input_data, 0), 0), 0)

        forward_transform = self.conv(
            Tensor(input_data, mstype.float32),
            Tensor(np.expand_dims(self.forward_basis, 1), mstype.float32),
        )

        forward_transform = forward_transform.asnumpy().squeeze(2)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
        phase = np.arctan2(imag_part, real_part)

        return magnitude, phase


class TacotronSTFT:
    """Tacotron."""
    def __init__(
            self,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            sampling_rate=22050,
            mel_fmin=0.0,
            mel_fmax=8000.0
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        self.mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )

    def spectral_normalize(self, x):
        """Normalize magnitudes."""
        output = np.log(np.clip(x, a_min=1e-5, a_max=np.max(x)))
        return output

    def mel_spectrogram(self, y):
        """
        Computes mel-spectrogram from wav.

        Args:
            y (np.array): Raw mel-spectrogram with shape (B, T) in range [-1, 1].

        Returns:
            mel_output (np.array): Mel-spectrogram  with shape (B, n_mel_channels, T).
        """
        magnitudes, _ = self.stft_fn.transform(y)
        mel_output = np.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)

        return mel_output
