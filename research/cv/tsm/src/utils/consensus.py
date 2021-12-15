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
"""consensuspictures"""
import mindspore.nn as nn


class Identity(nn.Cell):
    def construct(self, inp):
        return inp


class SegmentConsensus(nn.Cell):
    """SegmentConsensus"""
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        # self.shape = None

    def construct(self, input_tensor):
        # self.shape = input_tensor.shape
        if self.consensus_type == 'avg':
            output = input_tensor.mean(axis=self.dim, keep_dims=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(nn.Cell):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def construct(self, inp):
        return SegmentConsensus(self.consensus_type, self.dim)(inp)
