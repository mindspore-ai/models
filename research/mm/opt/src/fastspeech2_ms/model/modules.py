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
"""
fastspeech2 modules
"""
import os
import json
from collections import OrderedDict

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from fastspeech2_ms.utils.tools import get_mask_from_lengths, pad


def bucketize(input1, boundary, right=True):
    right = not right
    input_np = input1.asnumpy()
    boundary = boundary.asnumpy()
    output = np.digitize(input_np, boundary, right=right)
    output = Tensor(output)
    return output


class VarianceAdaptor(nn.Cell):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.exp = ops.Exp()
        self.linspace = ops.LinSpace()

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = ms.Parameter(
                self.exp(
                    self.linspace(Tensor(np.log(pitch_min), ms.float32), Tensor(np.log(pitch_max), ms.float32),
                                  n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = ms.Parameter(
                self.linspace(Tensor(pitch_min, ms.float32), Tensor(pitch_max, ms.float32), n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = ms.Parameter(
                self.exp(
                    self.linspace(Tensor(np.log(energy_min), ms.float32), Tensor(np.log(energy_max), ms.float32),
                                  n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = ms.Parameter(
                self.linspace(Tensor(energy_min, ms.float32), Tensor(energy_max, ms.float32), n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):
        """get_pitch_embedding"""
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        """get_energy_embedding"""
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def construct(
            self,
            x,
            src_mask,
            mel_mask=None,
            max_len=None,
            pitch_target=None,
            energy_target=None,
            duration_target=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):
        """VarianceAdapter construct"""
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = ops.clip_by_value(
                (self.round(self.exp(log_duration_prediction) - 1) * d_control),
                clip_value_min=Tensor(0, ms.float32), clip_value_max=Tensor(float('inf'), ms.float32)
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Cell):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.cat = ops.Concat(axis=0)

    def LR(self, x, duration, max_len):
        """LengthRegulator"""
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, Tensor(mel_len)

    def expand(self, batch, predicted):
        """LengthExpand"""
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].asnumpy().tolist()
            if expand_size == 0:
                continue
            size_0 = max(int(expand_size), 0)
            size_1 = vec.shape[0]
            broadcast_to = ops.BroadcastTo((size_0, size_1))
            vec = broadcast_to(vec)
            out.append(vec)
        out = self.cat(out)

        return out

    def construct(self, x, duration, max_len):
        """LengthRegulator construct"""
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Cell):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.SequentialCell(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm((self.filter_size,))),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm((self.filter_size,))),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Dense(self.conv_output_size, 1)

    # encoder_output: batch_size * seq_len * dim_size
    def construct(self, encoder_output, mask):
        """ construct"""
        out = self.conv_layer(encoder_output)

        out = self.linear_layer(out)

        # out: batch_size * seq_len
        out = out.squeeze(-1)

        # if mask is not None:
        #     out = out.masked_fill(mask.to(ms.bool_), 0.0)

        return out


class Conv(nn.Cell):
    """
    Convolution Module
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
            w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        # in_channels,
        # out_channels,
        # kernel_size,
        # stride = 1,
        # pad_mode = 'same',
        # padding = 0,
        # dilation = 1,
        # group = 1,
        # has_bias = False,
        # weight_init = 'normal',
        # bias_init = 'zeros'

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            has_bias=bias,
            pad_mode="pad",
        )

    def construct(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        x = x.transpose(0, 2, 1)

        return x
