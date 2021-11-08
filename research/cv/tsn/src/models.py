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
"""build net for tsn"""
import mindspore.nn as nn
import mindspore.ops as ops

from src.network import BNInception
from src.basic_ops import ConsensusModule

class TSN(nn.Cell):
    """create network"""
    def __init__(self, num_class, num_segments, modality,
                 base_model='BNInception', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8):
        super(TSN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type

        self.input_mean = [104, 117, 128]
        self.input_std = [1]
        self.input_size = 224

        self.Reshape = ops.Reshape()
        self.shape = ops.Shape()
        self.squeeze = ops.Squeeze(axis=1)
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        if self.modality == 'Flow':
            self.input_mean = [128]
        elif self.modality == 'RGBDiff':
            self.input_mean = self.input_mean * (1 + self.new_length)

        print(("""
        Initializing TSN with base model: {}.
        TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self.base_model = BNInception(num_class=self.num_class, modality=self.modality, dropout=self.dropout)
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def construct(self, x):
        """network compute"""
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length

        base_out = self.base_model(self.Reshape(x, (-1, sample_len) + self.shape(x)[-2:]))
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = self.Reshape(base_out, (-1, self.num_segments) + self.shape(base_out)[1:])

        output = self.consensus(base_out)
        output = self.squeeze(output)

        return output

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
