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

import mindspore as ms
from mindspore.nn import Cell, Dense
import mindspore.nn as nn


class PositionwiseFF(Cell):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if dropout == 0.0:
            self.CoreNet = nn.SequentialCell(
                Dense(d_model, d_inner), nn.ReLU(),
                Dense(d_inner, d_model),
            ).to_float(ms.float16)
        else:
            self.CoreNet = nn.SequentialCell(
                Dense(d_model, d_inner), nn.ReLU(),
                nn.Dropout(1 - dropout, dtype=ms.float16),
                Dense(d_inner, d_model),
                nn.Dropout(1 - dropout, dtype=ms.float16),
            ).to_float(ms.float16)
        self.layer_norm = nn.LayerNorm([d_model])
        self.pre_lnorm = pre_lnorm

    def construct(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))
            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            inp = self.cast(inp, ms.float16)
            core_out = self.CoreNet(inp)
            core_out = self.cast(core_out, ms.float32)
            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)
        return output
