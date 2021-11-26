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
######################## specific TCN model ########################
"""
import mindspore.nn as nn

from src.TCN import TemporalConvNet


class TCN(nn.Cell):
    """TCN"""

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dataset_name='permuted_mnist'):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Dense(num_channels[-1], output_size)
        self.logSoftmax = nn.LogSoftmax(axis=1) if dataset_name == 'permuted_mnist' else None

    def construct(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)
        o = self.linear(y1[:, :, -1])
        output = o if self.logSoftmax is None else self.logSoftmax(o)
        return output
