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

"""Ctc network definition."""

import numpy as np
from mindspore import Tensor, nn
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class CTCModel(nn.Cell):
    """Stacked Bidirectional LSTM
      Args:
            input_size(int): embedding size,39
            batch_size(int): batch_size.
            num_class(int): num of classed,62,61 is blank
            num_layers(int): layers of lstm
    """

    def __init__(self, input_size, batch_size, hidden_size, num_class, num_layers=2):
        super(CTCModel, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, num_layers=num_layers)
        self.embedding = nn.Dense(in_channels=2 * hidden_size, out_channels=num_class).to_float(mstype.float16)
        self.h0 = Tensor(np.zeros([2 * num_layers, batch_size, hidden_size]).astype(np.float32))
        self.c0 = Tensor(np.zeros([2 * num_layers, batch_size, hidden_size]).astype(np.float32))
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.shapes = P.Shape()
        self.cast = P.Cast()

    def construct(self, feature, masks):
        feature = self.transpose(feature, (1, 0, 2))
        masks = self.transpose(masks, (1, 0, 2))
        recurrent, _ = self.lstm(feature, (self.h0, self.c0))
        recurrent = self.cast(recurrent, mstype.float16)
        recurrent = recurrent * masks
        out = self.embedding(recurrent)
        out = self.cast(out, mstype.float32)
        return out
