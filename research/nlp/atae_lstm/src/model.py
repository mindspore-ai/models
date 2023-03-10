# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""AttentionLSTM Model"""
import numpy as np

import mindspore
from mindspore import nn, Parameter, Tensor
from mindspore import ops as P
from mindspore import ParameterTuple
from mindspore.common import dtype as mstype

from .model_utils.rnns import LSTM


class AttentionLstm(nn.Cell):
    """Model structure"""
    def __init__(self, config, weight, is_train=True):
        super(AttentionLstm, self).__init__()

        self.dim_word = config.dim_word
        self.dimh = config.dim_hidden
        self.dim_aspect = config.dim_aspect
        self.vocab_size = config.vocab_size
        self.grained = config.grained
        self.aspect_num = config.aspect_num
        self.embedding_table = weight
        self.is_train = is_train
        self.dropout_prob = config.dropout_prob
        self.batch_size = config.batch_size

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.mask = Tensor(np.random.uniform(size=(1, self.dimh)) > self.dropout_prob)

        self.embedding_word = nn.Embedding(vocab_size=self.vocab_size,
                                           embedding_size=self.dim_word,
                                           embedding_table=self.embedding_table)

        self.embedding_aspect = nn.Embedding(vocab_size=self.aspect_num,
                                             embedding_size=self.dim_aspect)

        self.dim_lstm_para = self.dim_word + self.dim_aspect

        self.init_state = Tensor(np.zeros((config.batch_size, 1, self.dimh)).astype(np.float32))

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.matmul = P.MatMul()
        self.expand = P.ExpandDims()
        self.cast = P.Cast()
        self.tanh = P.Tanh()
        self.tile = P.Tile()
        self.softmax_0 = P.Softmax(axis=0)
        self.softmax_1 = P.Softmax(axis=1)
        self.concat_0 = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)
        self.squeeze_0 = P.Squeeze(axis=0)
        self.squeeze_1 = P.Squeeze(axis=1)
        self.squeeze_2 = P.Squeeze(axis=2)
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.trans_matrix = (1, 0)

        u = lambda x: 1 / np.sqrt(x)
        e = u(self.dimh)
        self.w = Parameter(Tensor(np.zeros((self.dimh + self.dim_aspect, 1)).astype(np.float32)))
        self.ws = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.grained)).astype(np.float32)))
        self.bs = Parameter(Tensor(np.zeros((1, self.grained)).astype(np.float32)))
        self.wh = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float32)))
        self.wv = Parameter(Tensor(np.random.uniform(-e, e, (self.dim_aspect, self.dim_aspect)).astype(np.float32)))
        self.wp = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float32)))
        self.wx = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float32)))

        self.lstm = LSTM(self.dim_lstm_para, self.dimh, batch_first=True, has_bias=True)

        self.params = ParameterTuple((self.wv, self.wh, self.ws, self.bs, self.w, self.wp, self.wx))

    def construct(self, x, x_len, aspect):
        """
        shape:
            x: (1, N)  int32
            aspct: (1) int32
            x_len: (1) int32
        """
        # x: x shape: (B, N, 300)  aspect: (B, 300)
        x = self.embedding_word(x)
        aspect = self.embedding_aspect(aspect)

        x = self.cast(x, mstype.float32)
        aspect = self.cast(aspect, mstype.float32)

        # aspect: (B, N, 300)
        aspect = self.expand(aspect, 1)
        aspect_vector = self.tile(aspect, (1, x.shape[1], 1))

        lstm_input = self.concat_2((x, aspect_vector))

        h_0 = self.init_state
        c_0 = self.init_state

        output, (h_n, _) = self.lstm(lstm_input, (h_0, c_0), x_len)

        # h: [B, N, 300]  h_n [B, 1, 300]
        if self.batch_size == 1:
            h = self.squeeze_0(output)
            h_n = self.reshape(h_n, (1, 300))
        else:
            h = output

        if self.batch_size == 1:
            h_wh = self.matmul(h, self.wh)
            a_wv = self.matmul(self.reshape(aspect_vector, (-1, self.dim_aspect)), self.wv)
        else:
            h_wh = P.matmul(h, self.wh)
            a_wv = P.matmul(aspect_vector, self.wv)

        h_wh = self.cast(h_wh, mindspore.float32)
        a_wv = self.cast(a_wv, mindspore.float32)
        if self.batch_size == 1:
            m = self.tanh(self.concat_1((h_wh, a_wv)))
        else:
            m = self.tanh(self.concat_2((h_wh, a_wv)))

        m = self.cast(m, mindspore.float32)
        if self.batch_size == 1:
            tmp = self.matmul(m, self.w)
            tmp = self.reshape(tmp, (1, -1))
        else:
            tmp = P.matmul(m, self.w)
            tmp = self.squeeze_2(tmp)
            tmp = self.expand(tmp, 1)

        tmp = self.cast(tmp, mindspore.float32)
        alpha = self.softmax_1(tmp)
        alpha = self.cast(alpha, mindspore.float32)
        if self.batch_size == 1:
            r = self.matmul(alpha, h)
            r_wp = self.matmul(r, self.wp)
            h_wx = self.matmul(h_n, self.wx)
        else:
            r = P.matmul(alpha, h)
            r_wp = P.matmul(r, self.wp)
            h_wx = P.matmul(h_n, self.wx)
        r_wp = self.cast(r_wp, mindspore.float32)
        h_wx = self.cast(h_wx, mindspore.float32)
        h_star = self.tanh(r_wp + h_wx)
        # dropout
        if self.is_train:
            h_star = self.dropout(h_star)
        else:
            h_star = h_star * 0.5

        h_star = self.cast(h_star, mindspore.float32)
        if self.batch_size == 1:
            y_hat = self.matmul(h_star, self.ws) + self.bs
        else:
            y_hat = P.matmul(h_star, self.ws) + self.bs
        y_hat = self.cast(y_hat, mindspore.float32)
        if self.batch_size != 1:
            y_hat = self.squeeze_1(y_hat)
        y = self.softmax_1(y_hat)
        return y
