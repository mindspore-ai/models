# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mindspore
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import Uniform

from layers.LayerNorm import LayerNorm

class CausalLSTMCell(nn.Cell):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out, num_hidden_0,
                 input_shape, forget_bias=1.0, tln=False, init_value=0.001):
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple, the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        super(CausalLSTMCell, self).__init__()

        self.layer_name = layer_name
        self.filter_size = filter_size

        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.num_hidden_0 = num_hidden_0

        self.shape = input_shape
        self.batch = input_shape[0]
        self.height = input_shape[3]
        self.width = input_shape[4]

        self.layer_norm = tln
        self._forget_bias = forget_bias

        self.initializer = Uniform(init_value)

        self.zeros = P.Zeros()

        self.h_conv = nn.Conv2d(self.num_hidden, self.num_hidden*4, filter_size, 1, padding=0, pad_mode='same',
                                dilation=(1, 1), has_bias=True, weight_init=self.initializer)
        self.c_conv = nn.Conv2d(self.num_hidden, self.num_hidden*3, filter_size, 1, padding=0, pad_mode='same',
                                dilation=(1, 1), has_bias=True, weight_init=self.initializer)
        self.m_conv = nn.Conv2d(self.num_hidden_in, self.num_hidden*3, filter_size, 1, padding=0, pad_mode='same',
                                dilation=(1, 1), has_bias=True, weight_init=self.initializer)

        self.layer_norm_h = LayerNorm(self.num_hidden*4)
        self.layer_norm_c = LayerNorm(self.num_hidden*3)
        self.layer_norm_m = LayerNorm(self.num_hidden*3)

        if self.layer_name == 'lstm_1':
            self.x_conv = nn.Conv2d(self.shape[-3], self.num_hidden*7, filter_size, 1, padding=0, pad_mode='same',
                                    dilation=(1, 1), has_bias=True, weight_init=self.initializer)
        elif self.layer_name == 'lstm_2':
            self.x_conv = nn.Conv2d(self.num_hidden_0, self.num_hidden*7, filter_size, 1, padding=0, pad_mode='same',
                                    dilation=(1, 1), has_bias=True, weight_init=self.initializer)
        elif self.layer_name == 'lstm_3':
            self.x_conv = nn.Conv2d(self.num_hidden, self.num_hidden*7, filter_size, 1, padding=0, pad_mode='same',
                                    dilation=(1, 1), has_bias=True, weight_init=self.initializer)
        else:
            self.x_conv = nn.Conv2d(self.num_hidden, self.num_hidden*7, filter_size, 1, padding=0, pad_mode='same',
                                    dilation=(1, 1), has_bias=True, weight_init=self.initializer)

        self.layer_norm_x = LayerNorm(self.num_hidden*7)

        self.c2m_conv = nn.Conv2d(self.num_hidden, self.num_hidden*4, filter_size, 1, padding=0, pad_mode='same',
                                  dilation=(1, 1), has_bias=True, weight_init=self.initializer)

        self.m2o_conv = nn.Conv2d(self.num_hidden, self.num_hidden, filter_size, 1, padding=0, pad_mode='same',
                                  dilation=(1, 1), has_bias=True, weight_init=self.initializer)

        self.cell_conv = nn.Conv2d(self.num_hidden*2, self.num_hidden, 1, 1, padding=0, pad_mode='same',
                                   dilation=(1, 1), has_bias=True, weight_init='XavierUniform')

        self.layer_norm_c2m = LayerNorm(self.num_hidden*4)

        self.layer_norm_m2o = LayerNorm(self.num_hidden)

        self.split_4 = P.Split(1, 4)
        self.split_3 = P.Split(1, 3)
        self.split_7 = P.Split(1, 7)

        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

        self.concat = P.Concat(1)

    def construct(self, x, h, c, m):

        if h is None:
            h = self.zeros((self.batch, self.num_hidden, self.height, self.width), mindspore.float32)
        if c is None:
            c = self.zeros((self.batch, self.num_hidden, self.height, self.width), mindspore.float32)
        if m is None:
            m = self.zeros((self.batch, self.num_hidden_in, self.height, self.width), mindspore.float32)

        h_cc = self.h_conv(h)
        c_cc = self.c_conv(c)
        m_cc = self.m_conv(m)

        if self.layer_norm:
            h_cc = self.layer_norm_h(h_cc)
            c_cc = self.layer_norm_c(c_cc)
            m_cc = self.layer_norm_m(m_cc)

        i_h, g_h, f_h, o_h = self.split_4(h_cc)
        i_c, g_c, f_c = self.split_3(c_cc)
        i_m, f_m, m_m = self.split_3(m_cc)

        if x is None:

            i = self.sigmoid(i_h + i_c)
            f = self.sigmoid(f_h + f_c + self._forget_bias)
            g = self.tanh(g_h + g_c)
            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = None, None, None, None, None, None, None
            x_cc = None
        else:
            x_cc = self.x_conv(x)

            if self.layer_norm:
                x_cc = self.layer_norm_x(x_cc)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = self.split_7(x_cc)

            i = self.sigmoid(i_x + i_h + i_c)

            f = self.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = self.tanh(g_x + g_h + g_c)

        c_new = f * c + i * g

        c2m = self.c2m_conv(c_new)
        if self.layer_norm:
            c2m = self.layer_norm_c2m(c2m)
        i_c, g_c, f_c, o_c = self.split_4(c2m)

        if x is None:
            ii = self.sigmoid(i_c + i_m)
            ff = self.sigmoid(f_c + f_m + self._forget_bias)
            gg = self.tanh(g_c)
        else:
            ii = self.sigmoid(i_c + i_x_ + i_m)
            ff = self.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = self.tanh(g_c + g_x_)

        m_new = ff * self.tanh(m_m) + ii * gg

        o_m = self.m2o_conv(m_new)
        if self.layer_norm:
            o_m = self.layer_norm_m2o(o_m)
        if x is None:
            o = self.tanh(o_h + o_c + o_m)
        else:
            o = self.tanh(o_x + o_h + o_c + o_m)

        cell = self.concat((c_new, m_new))
        cell = self.cell_conv(cell)
        h_new = o * self.tanh(cell)
        return h_new, c_new, m_new
