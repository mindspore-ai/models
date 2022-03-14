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
"""Custom layers."""
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore.common.initializer import XavierUniform
from mindspore.common.initializer import initializer

from src.transformer.sublayers import MultiHeadAttention
from src.transformer.sublayers import PositionwiseFeedForward


class Linear(nn.Cell):
    """
    Create linear layer and init weights.
    """
    def __init__(
            self,
            in_dim,
            out_dim,
            bias=True,
            w_init='linear'
    ):
        super().__init__()

        if w_init == 'xavier':
            linear_weights = initializer(XavierUniform(), [in_dim, out_dim], mstype.float32)
        else:
            linear_weights = initializer(Normal(), [in_dim, out_dim], mstype.float32)

        self.linear_layer = nn.Dense(
            in_dim,
            out_dim,
            bias=bias,
            weight_init=linear_weights,
        )

    def construct(self, x):
        """Forward."""
        out = self.linear_layer(x)

        return out


class FFTBlock(nn.Cell):
    """
    Feed-forward transformer (FFT) block.
    Similar for 'encoder' and 'decoder' at this model.
    """
    def __init__(
            self,
            d_model,
            d_inner,
            n_head,
            d_k,
            d_v,
            dropout=0.1,
    ):
        super().__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def construct(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        """Forward"""
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output


class ConvNorm(nn.Cell):
    """
    Create convolution layer and init weights.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain='linear',
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        if w_init_gain == 'tanh':
            gain = 5.0 / 3
        else:
            gain = 1

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            weight_init=initializer(
                XavierUniform(gain=gain),
                [in_channels, out_channels],
                mstype.float32
            )
        )

    def construct(self, x):
        """Forward."""
        output = self.conv(x)

        return output
