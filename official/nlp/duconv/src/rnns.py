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
'''
rnns
'''
import math
import numpy as np
from mindspore import Parameter, Tensor, ParameterTuple
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops import constexpr
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype

@constexpr
def _init_state(shape, dtype):
    hx = Tensor(np.zeros(shape), dtype)
    return hx

class GRU(nn.Cell):
    '''
    GRU cell
    '''
    def __init__(self, input_size, hidden_size, has_bias=True, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        gate_size = 3 * hidden_size
        self.reverse = P.ReverseV2([0])
        self.reverse_sequence = P.ReverseSequence(0, 1)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.has_bias = has_bias
        self.gru = P.DynamicGRUV2()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions

        self.w_ih_list = []
        self.w_hh_list = []
        self.b_ih_list = []
        self.b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for direction in range(num_directions):
            suffix = '_reverse' if direction == 1 else ''
            self.w_ih_list.append(Parameter(Tensor(np.random.uniform(-stdv, stdv, (input_size, gate_size))
                                                   .astype(np.float32)), name='weight_ih{}'.format(suffix)))
            self.w_hh_list.append(Parameter(Tensor(np.random.uniform(-stdv, stdv, (hidden_size, gate_size))
                                                   .astype(np.float32)), name='weight_hh{}'.format(suffix)))
            if has_bias:
                self.b_ih_list.append(Parameter(Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                                                name='bias_ih{}'.format(suffix)))
                self.b_hh_list.append(Parameter(Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                                                name='bias_hh{}'.format(suffix)))
        self.w_ih_list = ParameterTuple(self.w_ih_list)
        self.w_hh_list = ParameterTuple(self.w_hh_list)
        self.b_ih_list = ParameterTuple(self.b_ih_list)
        self.b_hh_list = ParameterTuple(self.b_hh_list)

        self.cast = P.Cast()

    def construct(self, x, hx=None, seq_length=None):
        '''
        construct
        '''
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        if hx is None:
            hx = _init_state((self.num_directions, max_batch_size, self.hidden_size), x.dtype)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        if self.bidirectional:
            x, h = self._bi_dynamic_gru(x, hx, seq_length)
        else:
            x, h = self._dynamic_gru(x, hx, seq_length)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        return x, h

    def _dynamic_gru(self, x, hx, seq_length):
        '''
        dynamic_gru
        '''
        if self.has_bias:
            w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
                self.w_ih_list[0], self.w_hh_list[0], \
                self.b_ih_list[0], self.b_hh_list[0]
        else:
            w_f_ih, w_f_hh = self.w_ih_list[0], self.w_hh_list[0]
            b_f_ih, b_f_hh = None, None
        output_f, _, _, _, _, _ = self.gru(x, w_f_ih, w_f_hh, b_f_ih, b_f_hh, None, hx)
        h_f = output_f[-1]

        return output_f, h_f

    def _bi_dynamic_gru(self, x, hx, seq_length):
        '''
        dynamic_gru
        '''
        if self.has_bias:
            w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
                self.w_ih_list[0], self.w_hh_list[0], \
                self.b_ih_list[0], self.b_hh_list[0]
            w_b_ih, w_b_hh, b_b_ih, b_b_hh = \
                self.w_ih_list[1], self.w_hh_list[1], \
                self.b_ih_list[1], self.b_hh_list[1]
        else:
            w_f_ih, w_f_hh = self.w_ih_list[0], self.w_hh_list[0]
            w_b_ih, w_b_hh = self.w_ih_list[1], self.w_hh_list[1]
            b_f_ih, b_f_hh, b_b_ih, b_b_hh = None, None, None, None
        if seq_length is None:
            x_b = self.reverse(x)
        else:
            x_b = self.reverse_sequence(x, seq_length)
        output_f, _, _, _, _, _ = self.gru(x, w_f_ih, w_f_hh, b_f_ih, b_f_hh, None, hx[0])
        output_b, _, _, _, _, _ = self.gru(x_b, w_b_ih, w_b_hh, b_b_ih, b_b_hh, None, hx[1])
        if seq_length is None:
            output_b = self.reverse(output_b)
            h_f = output_f[-1]
            h_b = output_b[-1]
        else:
            output_b = self.reverse_sequence(output_b, seq_length)
            batch_index = mnp.arange(0, x.shape[1], 1, mstype.int32)
            indices = P.Concat()(batch_index, seq_length)
            h_f = P.GatherNd()(output_f, indices)
            h_b = P.GatherNd()(output_b, indices)
        hidden = P.Concat()((h_f, h_b))
        output = P.Concat(2)((output_f, output_b))
        return output, hidden.view(hx.shape)
