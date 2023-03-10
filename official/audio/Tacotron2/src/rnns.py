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
''' implement dynamic rnn'''
import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.ops.functional as F
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, Uniform
from mindspore import log as logger
from mindspore.ops.primitive import constexpr
from src.rnn_cells import rnn_relu_cell, rnn_tanh_cell, gru_cell
from src.rnn_cells import LSTMCell
from src.hparams import hparams as hps


@constexpr
def _init_state(shape, dtype, is_lstm):
    ''' init state '''
    hx = Tensor(np.zeros(shape), dtype)

    if is_lstm:
        ret = (hx, hx)
    else:
        ret = hx
    return ret


class DynamicRNN(nn.Cell):
    ''' dynamic rnn '''

    def __init__(self, mode):
        super(DynamicRNN, self).__init__()
        if mode == "RNN_RELU":
            cell = rnn_relu_cell
        elif mode == "RNN_TANH":
            cell = rnn_tanh_cell
        elif mode == "LSTM":
            cell = LSTMCell()
        elif mode == "GRU":
            cell = gru_cell
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        self.cell = cell
        self.is_lstm = mode == "LSTM"
        self.concat_len = 50
        self.pack = P.Stack()
        self.concat = P.Concat()
        self.squeeze = P.Squeeze()
        self.fp16_flag = hps.fp16_flag

    def pack_list(self, alignments):
        ''' pack tensor list '''
        align_tuple = ()
        n_frames = len(alignments)
        for i in range(n_frames // self.concat_len):
            start = i * self.concat_len
            end = (i + 1) * self.concat_len
            alignment = self.pack(alignments[start: end])
            align_tuple += (alignment,)
        if n_frames % self.concat_len != 0:
            start = n_frames // self.concat_len * self.concat_len
            alignment = self.pack(alignments[start:])
            align_tuple += (alignment,)
        alignments = self.concat(align_tuple)
        return alignments

    def recurrent(self, x, hidden, w_ih, w_hh, b_ih, b_hh):
        ''' static rnn '''
        time_step = range(x.shape[0])
        outputs = ()
        for t in time_step:
            hidden = self.cell(x[t], hidden, w_ih, w_hh, b_ih, b_hh)
            if self.is_lstm:
                outputs += (hidden[0],)
            else:
                outputs += (hidden,)
        outputs = self.pack_list(outputs)
        return outputs, hidden

    def variable_recurrent(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        ''' dynamic rnn '''
        time_step = range(x.shape[0])

        h_t = h
        if self.is_lstm:
            hidden_size = h[0].shape[-1]
            zero_output = P.ZerosLike()(h_t[0])
        else:
            hidden_size = h.shape[-1]
            zero_output = P.ZerosLike()(h_t)

        seq_length = P.BroadcastTo((hidden_size, -1))(seq_length)
        seq_length = P.Transpose()(seq_length, (1, 0))

        outputs = ()
        state_t = h_t
        for t in time_step:
            h_t = self.cell(self.squeeze(
                x[t:t + 1]), state_t, w_ih, w_hh, b_ih, b_hh)

            seq_cond = seq_length > t

            if self.is_lstm:
                if self.fp16_flag:
                    state_t_0 = P.Select()(seq_cond, F.cast(h_t[0], mstype.float16), state_t[0])
                    state_t_1 = P.Select()(seq_cond, F.cast(h_t[1], mstype.float16), F.cast(state_t[1], mstype.float16))
                    output = P.Select()(seq_cond, F.cast(h_t[0], mstype.float16), F.cast(zero_output, mstype.float16))
                else:
                    state_t_0 = P.Select()(seq_cond, h_t[0], state_t[0])
                    state_t_1 = P.Select()(seq_cond, h_t[1], state_t[1])
                    output = P.Select()(seq_cond, h_t[0], zero_output)
                state_t = (state_t_0, state_t_1)
            else:
                state_t = P.Select()(seq_cond, h_t, state_t)
                output = P.Select()(seq_cond, h_t, zero_output)
            outputs += (output,)

        outputs = self.pack_list(outputs)

        return outputs, state_t

    def construct(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        ''' rnn cells'''
        if seq_length is None:
            res = self.recurrent(x, h, w_ih, w_hh, b_ih, b_hh)
        else:
            res = self.variable_recurrent(
                x, h, seq_length, w_ih, w_hh, b_ih, b_hh)
        return res


class RNNBase(nn.Cell):
    ''' rnn base '''

    def __init__(
            self,
            mode,
            input_size,
            hidden_size,
            num_layers=1,
            has_bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False):
        super(RNNBase, self).__init__()
        if not 0 <= dropout <= 1:
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed")

        if dropout > 0 and num_layers == 1:
            logger.warning("dropout option adds dropout after all but last "
                           "recurrent layer, so non-zero dropout expects "
                           "num_layers greater than 1, but got dropout={} and "
                           "num_layers={}".format(dropout, num_layers))
        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        elif mode == "RNN_TANH":
            gate_size = hidden_size
        elif mode == "RNN_RELU":
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_op = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.has_bias = has_bias
        self.rnn = DynamicRNN(mode)
        self.squeeze = P.Squeeze()
        num_directions = 2 if bidirectional else 1
        self.is_lstm = mode == "LSTM"

        self.expand_dims = P.ExpandDims()
        self._all_weights = []
        self.w_list = []
        self.b_list = []

        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''
                offset = 2 ** (num_directions) * layer + 2 * direction
                w_ih = Parameter(Tensor(np.random.randn(gate_size, layer_input_size).astype(np.float32)),
                                 name='weight_ih_l{}{}'.format(layer, suffix))
                w_hh = Parameter(Tensor(np.random.randn(gate_size, hidden_size).astype(np.float32)),
                                 name='weight_hh_l{}{}'.format(layer, suffix))
                self.w_list.append(w_ih)
                self.w_list.append(w_hh)
                if has_bias:
                    b_ih = Parameter(Tensor(np.random.randn(gate_size).astype(np.float32)),
                                     name='bias_ih_l{}{}'.format(layer, suffix))
                    b_hh = Parameter(Tensor(np.random.randn(gate_size).astype(np.float32)),
                                     name='bias_hh_l{}{}'.format(layer, suffix))
                    self.b_list.append(b_ih)
                    self.b_list.append(b_hh)
                    layer_params = (self.w_list[offset],
                                    self.w_list[offset + 1],
                                    self.b_list[offset],
                                    self.b_list[offset + 1])

                else:
                    layer_params = (self.w_list[offset], self.w_list[offset + 1])

                self._all_weights.append(ParameterTuple(layer_params))
        self.w_list = ParameterTuple(self.w_list)
        self.b_list = ParameterTuple(self.b_list)
        self.reset_parameters()

    def reset_parameters(self):
        ''' init parameters '''
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))

    def _stacked_bi_dynamic_rnn(self, x, h, seq_length, weights):
        """stacked bidirectional dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        hidden, cell = h
        for i in range(self.num_layers):
            offset = i * 2
            if self.has_bias:
                w_f_ih, w_f_hh, b_f_ih, b_f_hh = weights[offset]
                w_b_ih, w_b_hh, b_b_ih, b_b_hh = weights[offset + 1]
            else:
                w_f_ih, w_f_hh = weights[offset]
                w_b_ih, w_b_hh = weights[offset + 1]
                b_f_ih, b_f_hh, b_b_ih, b_b_hh = None, None, None, None
            if self.is_lstm:
                h_f_i = (self.squeeze(hidden[offset:offset + 1]), self.squeeze(cell[offset:offset + 1]))
                h_b_i = (self.squeeze(hidden[offset + 1:]), self.squeeze(cell[offset + 1:]))
            else:
                h_f_i = self.squeeze(h[offset:offset + 1])
                h_b_i = self.squeeze(h[offset + 1:])

            if len(h_f_i[0].shape) <= 1:
                h_f_i = (self.expand_dims(h_f_i[0], 0), self.expand_dims(h_f_i[1], 0))
                h_b_i = (self.expand_dims(h_b_i[0], 0), self.expand_dims(h_b_i[1], 0))
            if seq_length is None:
                x_b = P.ReverseV2([0])(pre_layer)
            else:
                x_b = P.ReverseSequence(0, 1)(pre_layer, seq_length)
            output_f, h_t_f = self.rnn(pre_layer, h_f_i, seq_length, w_f_ih, w_f_hh, b_f_ih, b_f_hh)
            output_b, h_t_b = self.rnn(x_b, h_b_i, seq_length, w_b_ih, w_b_hh, b_b_ih, b_b_hh)
            hidden_f, cell_f = h_t_f
            hidden_b, cell_b = h_t_b
            if seq_length is None:
                output_b = P.ReverseV2([0])(output_b)
            else:
                output_b = P.ReverseSequence(0, 1)(output_b, seq_length)
            output = P.Concat(2)((output_f, output_b))
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (hidden_f, hidden_b,)
                c_n += (cell_f, cell_b,)
            else:
                h_n += (h_t_f, h_t_b,)
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(hidden.shape)
            c_n = c_n.view(cell.shape)

            return output, (h_n.view(hidden.shape), c_n.view(cell.shape))

        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def _stacked_dynamic_rnn(self, x, h, seq_length, weights):
        """stacked mutil_layer dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            if self.has_bias:
                w_ih, w_hh, b_ih, b_hh = weights[i]
            else:
                w_ih, w_hh = weights[i]
                b_ih, b_hh = None, None
            if self.is_lstm:
                h_i = (h[0][i], h[1][i])
            else:
                h_i = h[i]

            output, h_t = self.rnn(
                pre_layer, h_i, seq_length, w_ih, w_hh, b_ih, b_hh)
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t[0],)
                c_n += (h_t[1],)
            else:
                h_n += (h_t,)
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))

        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def construct(self, x, h=None, seq_length=None):
        ''' rnns '''
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        if h is None:
            h = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size),
                            x.dtype,
                            self.is_lstm)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        if self.bidirectional:
            x, h = self._stacked_bi_dynamic_rnn(x, h, seq_length, self._all_weights)
        else:
            x, h = self._stacked_dynamic_rnn(x, h, seq_length, self._all_weights)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        return x, h


class RNN(RNNBase):
    '''rnns '''

    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


class GRU(RNNBase):
    '''rnns '''

    def __init__(self, *args, **kwargs):
        mode = 'GRU'
        super(GRU, self).__init__(mode, *args, **kwargs)


class LSTM(RNNBase):
    '''rnns '''

    def __init__(self, *args, **kwargs):
        mode = 'LSTM'
        super(LSTM, self).__init__(mode, *args, **kwargs)
        self.support_non_tensor_inputs = True
