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
"""last layer for tsn"""
import mindspore.nn as nn
import mindspore.ops as ops

class ConsensusModule(nn.Cell):
    """TSN"""
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        if self.consensus_type == 'avg':
            output = self.mean(x, self.dim)
        elif self.consensus_type == 'identity':
            output = x
        else:
            output = None
        return output
