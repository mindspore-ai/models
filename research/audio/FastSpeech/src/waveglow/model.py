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
"""Model script."""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from src.waveglow.layers import Invertible1x1Conv
from src.waveglow.utils import fused_add_tanh_sigmoid_multiply


class WN(nn.Cell):
    """
    This is the WaveNet like layer for the affine coupling.
    The primary difference from WaveNet is the convolutions need not be causal.
    There is also no dilation size reset. The dilation only doubles on each layer.
    """
    def __init__(
            self,
            n_in_channels,
            n_mel_channels,
            n_layers,
            n_channels,
            kernel_size,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = nn.CellList()
        self.res_skip_layers = nn.CellList()

        self.start = nn.Conv1d(
            in_channels=n_in_channels,
            out_channels=n_channels,
            kernel_size=1,
            has_bias=True
        )

        self.end = nn.Conv1d(
            in_channels=n_channels,
            out_channels=2 * n_in_channels,
            kernel_size=1,
            has_bias=True
        )

        self.cond_layer = nn.Conv1d(
            in_channels=n_mel_channels,
            out_channels=2 * n_channels * n_layers,
            kernel_size=1,
            has_bias=True
        )

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)

            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels

            in_layer = nn.Conv1d(
                in_channels=n_channels,
                out_channels=2 * n_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_mode='pad',
                padding=padding,
                has_bias=True
            )

            res_skip_layer = nn.Conv1d(
                in_channels=n_channels,
                out_channels=res_skip_channels,
                kernel_size=1,
                has_bias=True
            )

            self.in_layers.append(in_layer)
            self.res_skip_layers.append(res_skip_layer)

        self.audio_zeros = Tensor(np.zeros((1, self.n_channels, 28800)), mstype.float32)

    def construct(self, audio, spect):
        """
        Forward.
        """
        audio = self.start(audio)
        output = self.audio_zeros

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i * 2 * self.n_channels

            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset: spect_offset + 2 * self.n_channels, :],
                self.n_channels
            )

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts

        output = self.end(output)

        return output


class WaveGlow(nn.Cell):
    """WaveGlow vocoder inference model."""
    def __init__(
            self,
            n_mel_channels,
            n_flows,
            n_group,
            n_early_every,
            n_early_size,
            wn_config,
            sigma=1.0
    ):
        super().__init__()

        self.upsample = nn.Conv1dTranspose(
            in_channels=n_mel_channels,
            out_channels=n_mel_channels,
            pad_mode='valid',
            kernel_size=1024,
            stride=256,
            has_bias=True,
        )

        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.wavenet = nn.CellList()
        self.convinv = nn.CellList()

        n_half = int(n_group / 2)
        n_remaining_channels = n_group
        audio_cells_list = []

        for k in range(n_flows):
            use_data_append = False
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
                use_data_append = True

            audio_cells_list.insert(
                0,
                AudioCell(
                    n_half=n_half,
                    n_mel_channels=n_mel_channels * n_group,
                    wn_config=wn_config,
                    use_data_append=use_data_append,
                    n_early_size=self.n_early_size,
                    sigma=sigma,
                    n_remaining_channels=n_remaining_channels,
                )
            )

        self.wavenet_blocks = nn.CellList(audio_cells_list)

        self.n_remaining_channels = n_remaining_channels

        self.concat = ops.Concat(axis=1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

        self.noise_shape = (1, self.n_remaining_channels, 28800)
        self.audio = Tensor(np.random.standard_normal(self.noise_shape), mstype.float32)

        self.time_cutoff = self.upsample.kernel_size[1] - self.upsample.stride[1]
        self.sigma = Tensor(sigma, mstype.float32)

    def construct(self, spect):
        """
        Forward to mel-spectrogram.

        Args:
            spect (Tensor): Mel-spectrogram. Shape (1, n_mel_channels, max_mel_len)

        Returns:
            audio (Tensor): Raw audio.
        """
        spect = self.upsample(spect)
        spect = spect[:, :, : - self.time_cutoff]
        bs, mel_size, channels = spect.shape

        spect = self.reshape(spect, (bs, mel_size, channels // self.n_group, self.n_group))
        spect = self.transpose(spect, (0, 2, 1, 3))
        spect = self.transpose(spect.view(spect.shape[0], spect.shape[1], -1), (0, 2, 1))

        audio = self.sigma * self.audio

        for audio_cell in self.wavenet_blocks:
            audio = audio_cell(audio, spect)

        audio = self.transpose(audio, (0, 2, 1)).view(audio.shape[0], -1)

        return audio


class AudioCell(nn.Cell):
    """Audio generator cell."""
    def __init__(
            self,
            n_half,
            n_mel_channels,
            wn_config,
            use_data_append,
            n_early_size,
            sigma,
            n_remaining_channels,
    ):
        super().__init__()
        self.n_half = n_half

        self.wn_cell = WN(n_half, n_mel_channels, **wn_config)
        self.convinv = Invertible1x1Conv(n_remaining_channels)
        self.sigma = Tensor(sigma, mstype.float32)

        self.use_data_append = bool(use_data_append)
        self.noise_shape = (1, n_early_size, 28800)

        self.z = Tensor(np.random.standard_normal(self.noise_shape), mstype.float32)
        self.concat = ops.Concat(axis=1)
        self.exp = ops.Exp()

    def construct(self, audio, spect):
        """Iterationaly restore audio from spectrogram."""
        audio_0 = audio[:, :self.n_half, :]
        audio_1 = audio[:, self.n_half:, :]

        output = self.wn_cell(audio_0, spect)

        s = output[:, self.n_half:, :]
        b = output[:, :self.n_half, :]

        audio_1 = (audio_1 - b) / self.exp(s)

        audio = self.concat((audio_0, audio_1))
        audio = self.convinv(audio)

        if self.use_data_append:
            z = self.z
            audio = self.concat((self.sigma * z, audio))

        return audio
