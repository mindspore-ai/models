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
"""Fastspeech2 Layers"""
import mindspore.nn as nn

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(nn.Cell):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def construct(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvNorm(nn.Cell):
    """Convolutional layer with normalization"""
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            has_bias=bias,
            pad_mode="pad"
        )

    def construct(self, signal):
        """ConvNorm construct"""
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Cell):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
            self,
            n_mel_channels=80,
            postnet_embedding_dim=512,
            postnet_kernel_size=5,
            postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.CellList()

        self.convolutions.append(
            nn.SequentialCell([
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
            ])
        )

        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.SequentialCell([
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                ])
            )

        self.convolutions.append(
            nn.SequentialCell([
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
            ])
        )
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    # x : (88, 1289, 80)
    def construct(self, x):
        """PostNet construct"""
        x = x.transpose((0, 2, 1))

        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
            x = self.tanh(x)
            x = self.dropout(x)

        x = self.convolutions[-1](x)
        x = self.dropout(x)
        x = x.transpose((0, 2, 1))

        return x
