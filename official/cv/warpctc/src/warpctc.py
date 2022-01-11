# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Warpctc network definition."""

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class StackedRNN(nn.Cell):
    """
     Define a stacked RNN network which contains two LSTM layers and one full-connect layer.

     Args:
        input_size(int): Size of time sequence. Usually, the input_size is equal to three times of image height for
        captcha images.
        batch_size(int): batch size of input data, default is 64
        hidden_size(int): the hidden size in LSTM layers, default is 512
        num_layer(int): the number of layer of LSTM.
     """

    def __init__(self, input_size, hidden_size=512, num_layer=2, batch_size=64):
        super(StackedRNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = 11
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer)

        self.fc_weight = np.random.random((self.num_classes, hidden_size)).astype(np.float32)
        self.fc_bias = np.random.random(self.num_classes).astype(np.float32)

        self.fc = nn.Dense(in_channels=hidden_size, out_channels=self.num_classes, weight_init=Tensor(self.fc_weight),
                           bias_init=Tensor(self.fc_bias))

        self.transpose = P.Transpose()

    def construct(self, x):
        x = self.transpose(x, (3, 0, 2, 1))
        x = self.reshape(x, (-1, self.batch_size, self.input_size))
        output, _ = self.lstm(x)
        res = self.fc(output)
        return res
