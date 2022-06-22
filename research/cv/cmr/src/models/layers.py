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

import mindspore.nn as nn


class FCBlock(nn.Cell):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Dense(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.SequentialCell(*module_list)

    def construct(self, x):
        x = self.fc_block(x)
        return x


class FCResBlock(nn.Cell):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.SequentialCell([
            nn.Dense(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.Dense(out_size, out_size),
            nn.BatchNorm1d(out_size)
        ])

        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(x + self.fc_block(x))
