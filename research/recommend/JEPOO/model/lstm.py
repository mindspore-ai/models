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
import mindspore.nn as nn


class BiLSTM(nn.Cell):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super(BiLSTM, self).__init__()
        self.input_size = input_features
        self.hidden_size = recurrent_features
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def construct(self, x):
        return self.rnn(x)[0]
