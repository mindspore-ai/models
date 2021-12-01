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
"""hypertext"""
import math
import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import Cell, SoftmaxCrossEntropyWithLogits, Dropout
from mindspore.ops import Fill, Squeeze, Concat
from src.mobius_linear import MobiusLinear
from src.poincare import EinsteinMidpoint, Logmap0


class HModel(Cell):
    """hypertext model"""

    def __init__(self, config):
        super(HModel, self).__init__()
        self.cat = Concat(axis=1)
        self.loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.config = config
        self.c_seed = 1.0
        self.fill = Fill()
        self.squeeze = Squeeze(1)
        self.min_norm = 1e-15
        num_input_fmaps = config.embed
        num_output_fmaps = config.n_vocab
        receptive_field_size = 1
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        std = 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
        emb = Tensor(np.random.normal(0, std, (config.n_vocab, config.embed)), mstype.float32)
        self.embedding = Parameter(emb, requires_grad=True)
        num_input_fmaps = config.embed
        num_output_fmaps = config.bucket
        receptive_field_size = 1
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        std = 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
        emb_wordngram = Tensor(np.random.normal(0, std, (config.bucket, config.embed)), mstype.float32)
        self.embedding_wordngram = Parameter(emb_wordngram)
        self.config_dropout = config.dropout
        if config.dropout != 0.0:
            self.dropout = Dropout(config.dropout)
        self.hyperLinear = MobiusLinear(config.embed,
                                        config.num_classes, c=self.c_seed)

        self.einstein_midpoint = EinsteinMidpoint(self.min_norm)
        self.logmap0 = Logmap0(self.min_norm)

    def construct(self, x1, x2):
        """class construction"""
        out_word = self.embedding[x1]
        out_wordngram = self.embedding_wordngram[x2]
        out = self.cat([out_word, out_wordngram])
        if self.config_dropout != 0.0:
            out = self.dropout(out)
        out = self.einstein_midpoint(out, c=self.c_seed)
        out = self.hyperLinear(out)
        out = self.logmap0(out, self.c_seed)
        return out
