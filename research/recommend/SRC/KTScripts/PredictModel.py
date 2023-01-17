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
from mindspore import nn, ops, numpy as mnp, ms_function

from KTScripts.BackModels import MLP, Transformer, CoKT


class PredictModel(nn.Cell):
    def __init__(self, feat_nums, embed_size, hidden_size, pre_hidden_sizes, dropout, output_size=1, with_label=True,
                 model_name='DKT'):
        super(PredictModel, self).__init__()
        self.item_embedding = nn.Embedding(feat_nums, embed_size)
        self.mlp = MLP(hidden_size, pre_hidden_sizes + [output_size], dropout=dropout)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.with_label = with_label
        self.move_label = True
        input_size_label = embed_size + 1 if with_label else embed_size
        self.model_name = model_name
        self.return_tuple = True
        if model_name == 'DKT':
            self.rnn = nn.LSTM(input_size_label, hidden_size, batch_first=True)
        elif model_name == 'Transformer':
            self.rnn = Transformer(input_size_label, hidden_size, dropout, head=4, b=1, position=True)
            self.return_tuple = False
        elif model_name == 'GRU4Rec':
            self.rnn = nn.GRU(input_size_label, hidden_size, batch_first=True)
            self.move_label = False

    def construct(self, x, y, mask=None):
        # x:(B, L,),y:(B, L)
        x = self.item_embedding(x)
        if self.with_label:
            if self.move_label:
                y_ = ops.concat((ops.zeros_like(y[:, 0:1]), y[:, :-1]), axis=1)
            else:
                y_ = y
            x = ops.concat((x, y_.expand_dims(-1)), axis=-1)
        o = self.rnn(x)
        if self.return_tuple:
            o = o[0]
        if mask is not None:
            o = ops.masked_select(o, ops.expand_dims(mask, -1))
            y = ops.masked_select(y, mask)
        o = o.reshape((-1, self.hidden_size))
        y = y.reshape(-1)
        o = self.mlp(o)
        if self.model_name == 'GRU4Rec':
            o = ops.softmax(o, -1)
        else:
            o = ops.sigmoid(o).squeeze(-1)
        return o, y

    @ms_function
    def learn_lstm(self, x, states1=None, states2=None, get_score=True):
        if states1 is None:
            states = None
        else:
            states = (states1, states2)
        return self.learn(x, states, get_score)

    def learn(self, x, states=None, get_score=True):
        x = self.item_embedding(x)  # (B, L, E)
        o = ops.zeros_like(x[:, 0:1, 0:1])  # (B, 1, 1)
        os = [None] * x.shape[1]
        with_label, rnn, mlp = self.with_label, self.rnn, self.mlp
        for i in range(x.shape[1]):
            x_i = x[:, i:i + 1]
            if with_label and get_score:
                x_i = ops.concat((x_i, o), -1)
            o, states = rnn(x_i, states)
            if get_score:
                o = ops.expand_dims(ops.sigmoid(mlp(o.squeeze(1))), 1)
            os[i] = o
        os = ops.concat(os, 1)  # (B, L) or (B, L, H)
        if self.output_size == 1:
            os = os.squeeze(-1)
        return os, states

    def GRU4RecSelect(self, origin_paths, n, skill_num, initial_logs):
        ranked_paths = [None] * n
        a1 = mnp.arange(origin_paths.shape[0]).expand_dims(-1)
        selected_paths = mnp.ones((origin_paths.shape[0], skill_num), dtype=mnp.bool_)
        selected_paths[a1, origin_paths] = False
        path, states = initial_logs, None
        a1 = a1.squeeze(-1)
        for i in range(n):
            o, states = self.learn(path, states)
            o = o[:, -1]
            o[selected_paths] = -1
            path = mnp.argmax(o, axis=-1)
            ranked_paths[i] = path
            selected_paths[a1, path] = True
            path = path.expand_dims(1)
        ranked_paths = ops.stack(ranked_paths, -1)
        return ranked_paths


class PredictRetrieval(PredictModel):
    def __init__(self, feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, with_label=True,
                 model_name='CoKT'):
        super(PredictRetrieval, self).__init__(feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, 1,
                                               with_label, model_name)
        if model_name == 'CoKT':
            self.rnn = CoKT(input_size + 1, hidden_size, dropout, head=2)

    def construct(self, intra_x, inter_his, inter_r, y, mask, inter_len):
        intra_x = self.item_embedding(intra_x)
        if self.with_label:
            y_ = ops.concat((ops.zeros_like(y[:, 0:1, None]), y[:, :-1, None]), axis=1).astype(mnp.float32)
            intra_x = ops.concat((intra_x, y_), axis=-1)
        inter_his = ops.concat((self.item_embedding(inter_his[:, :, :, 0]),
                                inter_his[:, :, :, 1:].astype(mnp.float32)), -1)
        inter_r = ops.concat((self.item_embedding(inter_r[:, :, :, 0]), inter_r[:, :, :, 1:].astype(mnp.float32)), -1)
        o = self.rnn(intra_x, inter_his, inter_r, mask, inter_len)
        o = ops.sigmoid(self.mlp(o)).squeeze(-1)
        y = ops.masked_select(y, mask).reshape(-1)
        return o, y

    def learn(self, intra_x, inter_his, inter_r, inter_len, states=None):
        his_len, seq_len = 0, intra_x.shape[1]
        intra_x = self.item_embedding(intra_x)  # (B, L, I)
        intra_h = None
        if states is not None:
            his_len = states[0].shape[1]
            intra_x = ops.concat((intra_x, states[0]), 1)  # (B, L_H+L, I)
            intra_h = states[1]
        o = mnp.zeros_like(intra_x[:, 0:1, 0:1])
        inter_his = ops.concat((self.item_embedding(inter_his[:, :, :, 0]),
                                inter_his[:, :, :, 1:].astype(mnp.float32)), 1)
        inter_r = ops.concat((self.item_embedding(inter_r[:, :, :, 0]), inter_r[:, :, :, 1:].astype(mnp.float32)), -1)
        M_rv, M_pv = self.rnn.deal_inter(inter_his, inter_r, inter_len)  # (B, L, R, H)
        os = []
        for i in range(seq_len):
            o, intra_h = self.rnn.step(M_rv[:, i], M_pv[:, i], intra_x[:, :i + his_len + 1], o, intra_h)
            o = ops.sigmoid(self.mlp(o))
            os.append(o)
        o = ops.concat(os, 1)  # (B, L, 1)
        return o, (intra_x, intra_h)


class ModelWithLoss(nn.Cell):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    @ms_function
    def construct(self, *data):
        output_data = self.model(*data)
        return self.criterion(*output_data), output_data

    @ms_function
    def output(self, *data):
        output_data = self.model(*data)
        return self.criterion(*output_data), output_data


class ModelWithLossMask(ModelWithLoss):
    def construct(self, *data):
        output_data = self.model(*data[:-1])
        return self.criterion(*output_data, data[-1]), output_data

    @ms_function
    def output(self, *data):
        output_data = self.model(*data[:-1])
        return self.criterion(*output_data, data[-1]), self.mask_fn(*output_data, data[-1].reshape(-1))

    @staticmethod
    def mask_fn(o, y, mask):
        o_mask = ops.masked_select(o, mask.expand_dims(-1)).reshape((-1, o.shape[-1]))
        y_mask = ops.masked_select(y, mask)
        return o_mask, y_mask


class ModelWithOptimizer(nn.Cell):
    def __init__(self, model_with_loss, optimizer, mask=False):
        super().__init__()
        self.mask = mask
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad_fn = ops.value_and_grad(self.model_with_loss, None, optimizer.parameters, has_aux=True)

    @ms_function
    def construct(self, *data):
        (loss, output_data), grads = self.grad_fn(*data)
        loss = ops.depend(loss, self.optimizer(grads))
        if self.mask:
            output_data = self.model_with_loss.mask_fn(*output_data, data[-1].reshape(-1))
        return loss, output_data
