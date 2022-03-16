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

        self.dropout = nn.Dropout(keep_prob=1 - self.dropout_prob)
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
        self.Ws = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.grained)).astype(np.float32)))
        self.bs = Parameter(Tensor(np.zeros((1, self.grained)).astype(np.float32)))
        self.Wh = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float32)))
        self.Wv = Parameter(Tensor(np.random.uniform(-e, e, (self.dim_aspect, self.dim_aspect)).astype(np.float32)))
        self.Wp = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float32)))
        self.Wx = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float32)))

        self.lstm = LSTM(self.dim_lstm_para, self.dimh, batch_first=True, has_bias=True)

        self.params = ParameterTuple((self.Wv, self.Wh, self.Ws, self.bs, self.w, self.Wp, self.Wx))

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

        # H: [B, N, 300]  h_n [B, 1, 300]
        H = output

        # Wh_H.size = (B, N, 300)
        H_Wh = P.matmul(H, self.Wh)
        # a_Wv.size = (B, N, 300)
        a_Wv = P.matmul(aspect_vector, self.Wv)
        # M.size = (B, N, 600)
        H_Wh = self.cast(H_Wh, mindspore.float32)
        a_Wv = self.cast(a_Wv, mindspore.float32)
        M = self.tanh(self.concat_2((H_Wh, a_Wv)))
        # tmp.size = (B, 1, N)
        M = self.cast(M, mindspore.float32)
        tmp = P.matmul(M, self.w)
        tmp = self.squeeze_2(tmp)
        tmp = self.expand(tmp, 1)
        # alpha.size = (B, N)
        tmp = self.cast(tmp, mindspore.float32)
        alpha = self.softmax_1(tmp)
        # r.size = (B, 1, 300)
        alpha = self.cast(alpha, mindspore.float32)
        r = P.matmul(alpha, H)

        r_Wp = P.matmul(r, self.Wp)
        h_Wx = P.matmul(h_n, self.Wx)
        # h_star.size = (B, 1, 300)
        r_Wp = self.cast(r_Wp, mindspore.float32)
        h_Wx = self.cast(h_Wx, mindspore.float32)
        h_star = self.tanh(r_Wp + h_Wx)
        # dropout
        if self.is_train:
            h_star = self.dropout(h_star)
        else:
            h_star = h_star * 0.5

        h_star = self.cast(h_star, mindspore.float32)
        y_hat = P.matmul(h_star, self.Ws) + self.bs
        y_hat = self.cast(y_hat, mindspore.float32)
        y_hat = self.squeeze_1(y_hat)
        y = self.softmax_1(y_hat)
        # y.size = (B, self.grained)
        return y
