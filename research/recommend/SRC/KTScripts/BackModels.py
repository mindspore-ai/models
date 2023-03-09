# Copyright 2023 Huawei Technologies Co., Ltd
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
from typing import Callable, Optional

import math
from mindspore import dtype as mdtype
from mindspore import nn, ms_function, numpy as mnp, ops, Parameter


class MLP(nn.SequentialCell):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: list[int],
                 norm_layer: Optional[Callable[..., nn.Cell]] = nn.BatchNorm1d,
                 activation_layer: Optional[Callable[..., nn.Cell]] = nn.LeakyReLU,
                 bias: bool = True,
                 dropout: float = 0.0):

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Dense(in_dim, hidden_dim, has_bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        layers.append(nn.Dense(in_dim, hidden_channels[-1], has_bias=bias))

        super().__init__(*layers)


@ms_function
def nll_loss(y_, y, mask):
    mask = mask.reshape(-1)
    return mnp.sum(ops.nll_loss(ops.log(y_ + 1e-9), y, reduction='none') * mask) / \
        mnp.sum(mask)


class MultiHeadedAttention(nn.Cell):
    def __init__(self, head, hidden_sizes, dropout_rate, input_sizes=None):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * 4
        if input_sizes is None:
            input_sizes = hidden_sizes
        for hidden_size in hidden_sizes:
            assert hidden_size % head == 0
        self.head = head
        self.head_size = hidden_sizes[0] // head
        self.hidden_size = hidden_sizes[-1]
        self.d_k = math.sqrt(hidden_sizes[0] // head)
        self.linear_s = nn.CellList(
            [nn.Dense(input_size, hidden_size) for (input_size, hidden_size) in zip(input_sizes, hidden_sizes)])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.batch_matmul = ops.BatchMatMul(transpose_b=True)

    @ms_function
    def attention(self, query, key, value, mask=None):
        scores = ops.div(self.batch_matmul(query, key), self.d_k)
        if mask is not None:
            scores = scores.masked_fill(ops.logical_not(mask), -1e9)
        p_attn = ops.softmax(scores, axis=-1)
        return ops.matmul(p_attn, value), p_attn

    def construct(self, query, key, value, mask=None):
        query, key, value = [
            l(x.view(-1, x.shape[-1])).view(*x.shape[:2], self.head, self.head_size).swapaxes(1, 2)
            for l, x in zip(self.linear_s, (query, key, value))]
        x, _ = self.attention(query, key, value, mask)  # (B, Head, L, D_H)
        x = x.swapaxes(1, 2)
        return self.linear_s[-1](x.view(-1, self.head * self.head_size)).view(*x.shape[:2], self.hidden_size)


class FeedForward(nn.Cell):
    def __init__(self, head, input_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.mh = MultiHeadedAttention(head, input_size, dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.activate = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm((input_size,))
        self.ln2 = nn.LayerNorm((input_size,))
        self.fc1 = nn.Dense(input_size, input_size)
        self.fc2 = nn.Dense(input_size, input_size)

    def construct(self, s, mask):
        s = s + self.dropout1(self.mh(s, s, s, mask))
        s = self.ln1(s)
        s_ = self.activate(self.fc1(s))
        s_ = self.dropout2(self.fc2(s_))
        s = self.ln2(s + s_)
        return s


class Transformer(nn.Cell):
    def __init__(self, input_size, hidden_size, dropout_rate, head=1, b=1, position=False, transformer_mask=True):
        super(Transformer, self).__init__()
        self.position = position
        if position:
            self.pe = PositionalEncoding(input_size, 0.5)
        self.fc = nn.Dense(input_size, hidden_size)
        self.SAs = nn.CellList([MultiHeadedAttention(head, hidden_size, dropout_rate) for _ in range(b)])
        self.FFNs = nn.CellList([FeedForward(head, hidden_size, dropout_rate) for _ in range(b)])
        self.b = b
        self.transformer_mask = transformer_mask

    def construct(self, inputs, mask=None):
        if self.position:
            inputs = self.pe(inputs)
        inputs = self.fc(inputs)
        max_len = inputs.shape[1]
        if self.transformer_mask:
            mask = mnp.tril(ops.ones((1, max_len, max_len), mdtype.bool_))
        elif mask is not None:
            mask = ops.expand_dims(mask, 1)  # (B, 1, L)
        if mask is not None:
            mask = ops.expand_dims(mask, 1)  # For head, shape is (B, 1, L, L) or (B, 1, 1, L)
        for i in range(self.b):
            inputs = self.SAs[i](inputs, inputs, inputs, mask)
            inputs = self.FFNs[i](inputs, mask)
        return inputs


class PositionalEncoding(nn.Cell):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = mnp.zeros((max_len, d_model))
        position = mnp.arange(0, max_len).expand_dims(1)
        div_term = ops.exp(mnp.arange(0, d_model, 2) *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)[:, :d_model // 2]
        pe = pe.expand_dims(0)
        self.pe = Parameter(pe, requires_grad=False)

    def construct(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class CoKT(nn.Cell):
    def __init__(self, input_size, hidden_size, dropout_rate, head=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.ma_inter = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            hidden_size + input_size - 1, hidden_size + input_size - 1, hidden_size + input_size, hidden_size))
        self.ma_intra = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            input_size - 1, input_size - 1, hidden_size + 1, hidden_size))
        self.wr = Parameter(mnp.randn(1, 1, 2))
        self.ln = nn.Dense(2 * hidden_size + input_size - 1, hidden_size)

    def construct(self, intra_x, inter_his, inter_r, intra_mask, inter_len):
        # (B, L, I), (B, L*R, L, I), (B, L, R, I), (B, L), (B, L, R)
        intra_mask = intra_mask.expand_dims(-1)  # (B, L, 1)

        intra_h, _ = self.rnn(intra_x)  # (B, L, H)
        intra_h_mask = ops.masked_select(intra_h, intra_mask).view(-1, self.hidden_size)  # (seq_sum, H)
        intra_x_mask = ops.masked_select(intra_x, intra_mask).view(-1, self.input_size)  # (seq_sum, I)
        # inter attention
        intra_mask_ = intra_mask.expand_dims(-1)  # (B, L, 1, 1)
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.shape[0] * inter_his.shape[1], *inter_his.shape[2:]))  # (B*L*R, L, H)
        inter_his = inter_his[mnp.arange(inter_his.shape[0]), inter_len.view(-1) - 1]  # (B*L*R, H)
        inter_his = inter_his.view(*inter_len.shape, self.hidden_size)  # (B, L, R, H)
        inter_his = ops.masked_select(inter_his, intra_mask_).view(-1, *inter_his.shape[2:])  # (seq_sum, R, H)
        inter_r = ops.masked_select(inter_r, intra_mask_).view(-1, *inter_r.shape[2:])  # (seq_sum, R, I)
        M_rv = ops.concat((inter_his, inter_r), -1).view(
            *inter_r.shape[:2], self.hidden_size + self.input_size)  # (seq_sum, R, H+I)
        M_pv = M_rv[:, :, :-1].view(*M_rv.shape[:2], self.input_size + self.hidden_size - 1)  # (seq_sum, R, H+I-1)
        m_pv = ops.concat((intra_h_mask, intra_x_mask[:, :-1]), 1).view(
            M_pv.shape[0], 1, self.hidden_size + self.input_size - 1)  # (seq_sum, 1, H+I-1)
        v_v = self.ma_inter(m_pv, M_pv, M_rv).squeeze(1)  # (seq_sum, H)
        # intra attention
        intra_x_p = intra_x[:, :, :-1]  # (B, L, I-1)
        intra_h_p = ops.concat((intra_h, intra_x[:, :, -1:]), -1)  # (B, L, H+1)
        intra_mask_attn = mnp.tril(mnp.ones((1, 1, intra_x_p.shape[1], intra_x_p.shape[1]), mnp.bool_))
        v_h = self.ma_intra(intra_x_p, intra_x_p, intra_h_p, mask=intra_mask_attn)  # (B, L, H)
        v_h = ops.masked_select(v_h, intra_mask).view(-1, v_h.shape[-1])  # (seq_sum, H)
        v = mnp.sum(ops.softmax(self.wr, -1) * ops.stack((v_v, v_h), -1), -1)  # (seq_sum, H)
        return self.ln(ops.concat((v, intra_h_mask, intra_x_mask[:, :-1]), 1))  # (seq_sum, H)

    def deal_inter(self, inter_his, inter_r, inter_len):
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.shape[0] * inter_his.shape[1], *inter_his.shape[2:]))  # (B*L*R, L, H)
        inter_his = inter_his[mnp.arange(inter_his.shape[0]), inter_len.view(-1) - 1]  # (B*L*R, H)
        inter_his = inter_his.view(*inter_len.shape, self.hidden_size)  # (B, L, R, H)
        M_rv = ops.concat((inter_his, inter_r), -1).view(
            *inter_r.shape[:3], self.hidden_size + self.input_size)  # (B, L, R, H+I)
        M_pv = M_rv[:, :, :-1].view(*M_rv.shape[:3], self.input_size + self.hidden_size - 1)  # (B, L, R, H+I)
        return M_rv, M_pv

    def step(self, m_rv, M_pv, intra_x, o, intra_h_p=None):
        # M_*: (B, R, H)
        # intra_h_p:(B, L-1, H+1), with the y
        # intra_x:(B, L, I-1), without the y
        # o: y from last step
        intra_h_next, _ = self.rnn(ops.concat((intra_x[:, -1:], o), axis=-1),
                                   None if intra_h_p is None else intra_h_p[:, -1, :-1].expand_dims(0))  # (B, 1, H)
        m_pv = ops.concat((intra_h_next, intra_x[:, -1:]), -1)  # (B, 1, H+I-1)
        v_v = self.ma_inter(m_pv, M_pv, m_rv)  # (B, 1, H)

        intra_x_p = intra_x
        intra_h_next = ops.concat((intra_h_next, o), axis=-1)
        intra_h_p = intra_h_next if intra_h_p is None else ops.concat((intra_h_p, intra_h_next), 1)  # (B, L, H+1)
        # Sequence mask
        v_h = self.ma_intra(intra_x_p[:, -1:], intra_x_p, intra_h_p)  # (B, 1, H), only query last target item
        v = mnp.sum(ops.softmax(self.wr, -1) * mnp.stack((v_v, v_h), -1), -1)  # (B, 1, H)
        return self.l(ops.concat((v, intra_h_p[:, -1:, :-1], intra_x[:, -1:]), -1)), intra_h_p  # (B, 1, 2*H+I-1)


if __name__ == '__main__':
    from mindspore import context
    import time

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    model = CoKT(16, 14, 0.5)
    seq_len_ = list(range(100, 139))
    max_len_ = 200
    for j in range(len(seq_len_)):
        t0 = time.perf_counter()
        seq_len = seq_len_[:j + 1]
        seq_sum = sum(seq_len)
        mask_ = ops.sequence_mask(ops.Tensor(seq_len), max_len_)
        x_ = mnp.rand(len(seq_len), max_len_, 16)
        his = mnp.rand(len(seq_len), max_len_ * 5, max_len_, 16)
        r = mnp.rand(len(seq_len), max_len_, 5, 16)
        inter_len_ = mnp.randint(1, 200, r.shape[:3])
        t1 = time.perf_counter()
        print(t1 - t0)
        output = model(x_, his, r, mask_, inter_len_)
        ops.print_(output)
        print(time.perf_counter() - t1)
