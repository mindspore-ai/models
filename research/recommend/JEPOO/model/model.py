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
import mindspore as ms
from mindspore import ops, nn
from mindspore import Tensor
from .lstm import BiLSTM
from .mel import MelSpectrogram


class BasicBlock(nn.Cell):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.sample = nn.SequentialCell([
            nn.Conv2d(inplanes, planes, (1, 1), has_bias=True),
            nn.BatchNorm2d(planes)
        ])
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(inplanes, planes, (3, 3), pad_mode='same', has_bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        ])
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(planes, planes, (3, 3), pad_mode='same', has_bias=True),
            nn.BatchNorm2d(planes),
        ])
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.sample(x)
        out = self.relu(out)
        return out


class ConvStackS(nn.Cell):
    def __init__(self, model_size):
        super().__init__()
        self.cnn = nn.SequentialCell([
            BasicBlock(1, model_size * 4),
            nn.Dropout(0.25),
            BasicBlock(model_size * 4, model_size * 6),
            nn.Dropout(0.25),
            BasicBlock(model_size * 6, model_size * 8),
            nn.Dropout(0.25)
        ])

    def construct(self, mel):
        x = ops.expand_dims(mel, 1)
        x = self.cnn(x)
        return x


class ConvStack(nn.Cell):
    def __init__(self, input_features, model_size):
        super().__init__()
        self.cnn = nn.SequentialCell(
            BasicBlock(model_size * 8, model_size * 3),
            nn.Dropout(0.25),
            BasicBlock(model_size * 3, model_size * 4),
            nn.Dropout(0.25),
            BasicBlock(model_size * 4, model_size * 6),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.25),
            BasicBlock(model_size * 6, model_size * 8),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.SequentialCell([
            nn.Dense((model_size * 8) * (input_features // 4), model_size * 48),
            nn.Dropout(0.5)
        ])

    def construct(self, x):
        x = self.cnn(x)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.fc(x)
        return x


class JM(nn.Cell):
    def __init__(self, input_features, output_features, sample_rate, hop_length, mel_fmin, mel_fmax, model_size=16):
        super().__init__()

        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.melspectrogram = MelSpectrogram(input_features, sample_rate, hop_length * 4, hop_length, mel_fmin=mel_fmin,
                                             mel_fmax=mel_fmax)
        self.cnn_s = ConvStackS(model_size)

        self.onset_stack = nn.SequentialCell([
            ConvStack(input_features, model_size),
            sequence_model(model_size * 48, model_size * 48),
        ])
        self.onset_fc = nn.SequentialCell([
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

        self.offset_stack = nn.SequentialCell([
            ConvStack(input_features, model_size),
            sequence_model(model_size * 48, model_size * 48),
        ])

        self.offset_fc = nn.SequentialCell([
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

        self.frame_stack = nn.SequentialCell([
            ConvStack(input_features, model_size),
            sequence_model(model_size * 48, model_size * 48),
        ])

        self.frame_fc = nn.SequentialCell([
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

        self.combined_stack = nn.SequentialCell([
            sequence_model(output_features * 3, model_size * 48),
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

    def construct(self, audio):
        mel = self.melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = ops.transpose(mel, (0, 2, 1))
        x = self.cnn_s(mel)
        onset_act = self.onset_stack(x)
        onset_pred = self.onset_fc(onset_act)
        offset_act = self.offset_stack(x)
        offset_pred = self.offset_fc(offset_act)
        activation_pred = self.frame_stack(x)
        frame_act = self.frame_fc(activation_pred)
        combined_pred = ops.Concat(-1)([onset_pred, frame_act, offset_pred])
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, frame_pred


class JMPML(nn.Cell):
    def __init__(self, input_features, output_features, sample_rate, hop_length, mel_fmin, mel_fmax, model_size=16):
        super().__init__()

        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.melspectrogram = MelSpectrogram(input_features, sample_rate, hop_length * 4, hop_length, mel_fmin=mel_fmin,
                                             mel_fmax=mel_fmax)
        self.cnn_s = ConvStackS(model_size)

        self.onset_stack = nn.SequentialCell([
            ConvStack(input_features, model_size),
            sequence_model(model_size * 48, model_size * 48),
        ])
        self.onset_fc = nn.SequentialCell([
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

        self.offset_stack = nn.SequentialCell([
            ConvStack(input_features, model_size),
            sequence_model(model_size * 48, model_size * 48),
        ])

        self.offset_fc = nn.SequentialCell([
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

        self.frame_stack = nn.SequentialCell([
            ConvStack(input_features, model_size),
            sequence_model(model_size * 48, model_size * 48),
        ])

        self.frame_fc = nn.SequentialCell([
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])

        self.combined_stack = nn.SequentialCell([
            sequence_model(output_features * 3, model_size * 48),
            nn.Dense(model_size * 48, output_features),
            nn.Sigmoid()
        ])
        self.pml_nn = nn.SequentialCell([
            nn.Dense(3, 3, weight_init=Tensor(np.eye(3), dtype=ms.float32)),
            nn.Softmax()
        ])

    def construct(self, audio, w_tasks=None):
        mel = self.melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = ops.transpose(mel, (0, 2, 1))
        x = self.cnn_s(mel)
        onset_act = self.onset_stack(x)
        onset_pred = self.onset_fc(onset_act)
        offset_act = self.offset_stack(x)
        offset_pred = self.offset_fc(offset_act)
        activation_pred = self.frame_stack(x)
        frame_act = self.frame_fc(activation_pred)
        combined_pred = ops.Concat(-1)([onset_pred, frame_act, offset_pred])
        frame_pred = self.combined_stack(combined_pred)
        if w_tasks is None:
            return onset_pred, offset_pred, frame_pred
        return [onset_pred, offset_pred, frame_pred, 3 * self.pml_nn(w_tasks)]
