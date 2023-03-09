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
"""Model modules."""
from collections import OrderedDict

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as msnp
from mindspore import ops
from mindspore.common.initializer import XavierUniform
from mindspore.common.initializer import initializer

from src.cfg.config import config as hp


class LengthRegulator(nn.Cell):
    """
    Length Regulator.

    Predicts duration of the each phonem,
    and let change speed of speech with alpha.
    """
    def __init__(self):
        super().__init__()
        self.duration_predictor = DurationPredictor()

        self.tile = ops.Tile()
        self.round = ops.Round()
        self.stack = ops.Stack()
        self.zeros = ops.Zeros()
        self.concat = ops.Concat()
        self.matmul = ops.MatMul()
        self.sum = ops.ReduceSum()
        self.bmm = ops.BatchMatMul()
        self.unsqueeze = ops.ExpandDims()
        self.max = ops.ArgMaxWithValue(axis=-1)
        self.mesh = ops.Meshgrid(indexing='xy')

        self.alignment_zeros = self.zeros(
            (hp.batch_size, hp.mel_max_length, hp.character_max_length),
            mstype.float32,
        )

        # For alignment
        self.h = hp.mel_max_length
        self.w = hp.character_max_length
        self.base_mat_ones = msnp.ones((self.h, self.w))
        self.meshgrid = self.mesh((msnp.arange(self.w), msnp.arange(self.h)))[1]
        self.zero_tensor = Tensor([0.])
        self.mel_pos_linspace = self.unsqueeze(msnp.arange(hp.mel_max_length) + 1, 0)

    def LR(self, enc_out, duration_predictor_output, mel_max_length=None):
        """Length regulator module."""
        expand_max_len = self.sum(duration_predictor_output.astype(mstype.float32))

        # None during eval
        if mel_max_length is not None:
            alignment = self.alignment_zeros
        else:
            alignment = self.unsqueeze(self.alignment_zeros[0], 0)

        for i in range(duration_predictor_output.shape[0]):
            thresh_2 = duration_predictor_output[i].cumsum().astype(mstype.float32)
            thresh_1 = self.concat(
                (
                    self.zero_tensor.astype(mstype.float64),
                    thresh_2[:-1].astype(mstype.float64)
                )
            )
            thresh_1 = self.tile(thresh_1, (self.h, 1))
            thresh_2 = self.tile(thresh_2, (self.h, 1))

            low_thresh = (self.meshgrid < thresh_2).astype(mstype.float32)
            up_thresh = (self.meshgrid >= thresh_1).astype(mstype.float32)
            intersection = low_thresh * up_thresh
            res = intersection.astype(mstype.bool_)
            alignment[i] = msnp.where(res, self.base_mat_ones, alignment[i])

        output = self.bmm(alignment, enc_out)

        return output, expand_max_len

    def construct(self, encoder_output, alpha=1.0, target=None, mel_max_length=None):
        """
        Predict duration of each phonema.
        """
        duration_predictor_output = self.duration_predictor(encoder_output)

        # Not none during training
        if target is not None:
            output, _ = self.LR(encoder_output, target, mel_max_length=mel_max_length)

            return output, duration_predictor_output

        duration_predictor_output = (duration_predictor_output + 0.5) * alpha
        duration_predictor_output = self.round(duration_predictor_output.copy())

        output, mel_len = self.LR(encoder_output, duration_predictor_output)

        mel_pos_mask = (self.mel_pos_linspace <= mel_len).astype(mstype.float32)
        mel_pos = self.mel_pos_linspace * mel_pos_mask

        return output, mel_pos, mel_len


class DurationPredictor(nn.Cell):
    """
    Duration Predictor.

    Predicts duration of the each phonem.
    """
    def __init__(self):
        super().__init__()

        self.input_size = hp.encoder_dim
        self.filter_size = hp.duration_predictor_filter_size
        self.kernel = hp.duration_predictor_kernel_size
        self.conv_output_size = hp.duration_predictor_filter_size
        self.dropout = 1 - hp.dropout

        self.conv_layer = nn.SequentialCell(OrderedDict([
            ("conv1d_1", Conv(
                self.input_size,
                self.filter_size,
                kernel_size=self.kernel,
                padding=1)),
            ("layer_norm_1", nn.LayerNorm([self.filter_size])),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(p=1 - self.dropout)),
            ("conv1d_2", Conv(
                self.filter_size,
                self.filter_size,
                kernel_size=self.kernel,
                padding=1)),
            ("layer_norm_2", nn.LayerNorm([self.filter_size])),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(p=1 - self.dropout))
        ]))

        self.linear_layer = nn.Dense(
            in_channels=self.conv_output_size,
            out_channels=1,
            weight_init=initializer(
                XavierUniform(),
                [1, self.conv_output_size],
                mstype.float32
            )
        )

        self.relu = nn.ReLU()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()

    def construct(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = self.relu(out)
        out = self.squeeze(out)

        if not self.training:
            out = self.expand_dims(out, 0)

        return out


class BatchNormConv1d(nn.Cell):
    """
    Custom BN, Conv1d layer with weight init.
    """
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size,
            stride,
            padding,
            activation=None,
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            has_bias=False,
            weight_init=initializer(
                XavierUniform(),
                [out_dim, in_dim, kernel_size],
                mstype.float32,
            )
        )

        self.bn = nn.BatchNorm2d(out_dim, use_batch_statistics=True)

        self.activation = activation
        self.expand_dims = ops.ExpandDims()

    def construct(self, input_tensor):
        out = self.conv1d(input_tensor)

        if self.activation is not None:
            out = self.activation(out)

        out = self.bn(self.expand_dims(out, -1))
        out = out.squeeze(-1)

        return out


class Conv(nn.Cell):
    """
    Conv1d with weight init.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            dilation=dilation,
            has_bias=bias,
            weight_init=initializer(
                XavierUniform(),
                [in_channels, out_channels, kernel_size],
                mstype.float32,
            )
        )

        self.transpose = ops.Transpose()

    def construct(self, x):
        x = self.transpose(x, (0, 2, 1))
        x = self.conv(x)
        x = self.transpose(x, (0, 2, 1))

        return x


class Highway(nn.Cell):
    """Highway network."""
    def __init__(self, in_size, out_size):
        super().__init__()
        self.h = nn.Dense(in_size, out_size, bias_init='zeros')
        self.t = nn.Dense(in_size, out_size, bias_init=Tensor(np.full(in_size, -1.), mstype.float32))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, inputs):
        out_1 = self.relu(self.h(inputs))
        out_2 = self.sigmoid(self.t(inputs))
        output = out_1 * out_2 + inputs * (1.0 - out_2)

        return output


class CBHG(nn.Cell):
    """
    CBHG a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """
    def __init__(self, in_dim, num_banks, projections):
        super().__init__()
        self.in_dim = in_dim

        self.relu = nn.ReLU()
        self.conv1d_banks = nn.CellList(
            [
                BatchNormConv1d(
                    in_dim,
                    in_dim,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    activation=self.relu,
                )
                for k in range(1, num_banks + 1)
            ]
        )

        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, pad_mode='same')

        in_sizes = [num_banks * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]

        self.conv1d_projections = nn.CellList(
            [
                BatchNormConv1d(
                    in_size,
                    out_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=activation,
                )
                for (in_size, out_size, activation) in zip(in_sizes, projections, activations)
            ]
        )

        self.highways = nn.CellList([Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(in_dim, in_dim, 1, batch_first=True, bidirectional=True)

        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=1)

    def construct(self, inputs):
        """
        Forward mels to recurrent network.
        """
        out = self.transpose(inputs, (0, 2, 1))

        last_dim = out.shape[-1]

        output_list = []
        for conv in self.conv1d_banks:
            output_list.append(conv(out)[:, :, :last_dim])

        output = self.concat(output_list)
        output = self.max_pool1d(output)[:, :, :last_dim]

        for conv1d in self.conv1d_projections:
            output = conv1d(output)

        output = self.transpose(output, (0, 2, 1))
        output += inputs

        for highway in self.highways:
            output = highway(output)

        outputs, _ = self.gru(output)

        return outputs
