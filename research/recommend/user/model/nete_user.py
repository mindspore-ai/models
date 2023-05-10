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

import copy
from mindspore import nn, ops
import mindspore.numpy as np
from mindspore.common.initializer import Uniform


# Paper Reference: Lei Li et al."Generate Neural Template Explanations for Recommendation." in CIKM 2020.
# Code Reference: https://github.com/lileipisces/NLG4RS


class GFRU(nn.Cell):
    def __init__(self, hidden_size):
        super(GFRU, self).__init__()

        self.layer_w = nn.Dense(hidden_size, hidden_size)
        self.layer_f = nn.Dense(hidden_size, hidden_size)
        self.layer_w_f = nn.Dense(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def construct(self, state_w, state_f):
        state_w_ = self.layer_w(state_w)  # (1, batch_size, rnn_dim) -> ((1, batch_size, rnn_dim))
        state_w_ = self.tanh(state_w_)
        state_f_ = self.layer_f(state_f)  # (1, batch_size, rnn_dim) -> ((1, batch_size, rnn_dim))
        state_f_ = self.tanh(state_f_)
        state_w_f = ops.concat([state_w_, state_f_], 2)  # (1, batch_size, hidden_dim*2)
        gamma = self.layer_w_f(state_w_f)
        gamma = self.sigmoid(gamma)  # (1, batch_size, 1)

        return gamma


class NeteUser(nn.Cell):
    def __init__(self, nuser, nitem, ntoken, nfeature, emsize, rnn_dim, dropout_prob, hidden_size, t,
                 num_layers=2, nsentiment=2):
        super(NeteUser, self).__init__()
        # model embedding
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        # text task
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.sentiment_embeddings = nn.Embedding(nsentiment, emsize)

        self.gru_w = nn.GRU(emsize, rnn_dim, batch_first=True)
        self.gru_f = nn.GRU(emsize, rnn_dim, batch_first=True)
        self.gfru = GFRU(rnn_dim)
        self.predict_linear = nn.Dense(rnn_dim, ntoken, weight_init=Uniform(0.1))
        self.dropout = nn.Dropout(keep_prob=dropout_prob)

        self.tanh = nn.Tanh()
        self.trans_linear = nn.Dense(emsize * 5, rnn_dim, weight_init=Uniform(0.1))

        # rating task
        self.first_layer = nn.Dense(emsize * 3, hidden_size, weight_init=Uniform(0.1))
        layer = nn.Dense(hidden_size, hidden_size, weight_init=Uniform(0.1))
        self.layers = nn.SequentialCell([copy.deepcopy(layer) for _ in range(num_layers)])
        self.last_layer = nn.Dense(hidden_size, 1, weight_init=Uniform(0.1))
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(axis=-1)
        self.nitem = nitem
        self.nfeature = nfeature

        # ips_embedding
        self.user_embeddings_mlp = nn.Embedding(nuser, emsize)
        self.item_embeddings_mlp = nn.Embedding(nitem, emsize)
        self.feature_embeddings_mlp = nn.Embedding(nfeature, emsize)
        self.softmax = nn.Softmax(axis=0)
        self.t = t

        # confounder modeling MLP
        self.confounder1_layer = nn.SequentialCell(
            nn.Dense(emsize * 2, hidden_size, weight_init=Uniform(0.1)),
            nn.Dropout(keep_prob=dropout_prob),
            nn.ReLU(),
            nn.Dense(hidden_size, emsize, weight_init=Uniform(0.1))
        )
        self.confounder2_layer = nn.SequentialCell(
            nn.Dense(emsize * 2, hidden_size, weight_init=Uniform(0.1)),
            nn.Dropout(keep_prob=dropout_prob),
            nn.ReLU(),
            nn.Dense(hidden_size, emsize, weight_init=Uniform(0.1))
        )
        self.confounder3_layer = nn.SequentialCell(
            nn.Dense(emsize * 3, hidden_size, weight_init=Uniform(0.1)),
            nn.Dropout(keep_prob=dropout_prob),
            nn.ReLU(),
            nn.Dense(hidden_size, emsize, weight_init=Uniform(0.1))
        )

        # treatment prediction
        self.treat1_layer = nn.SequentialCell(
            nn.Dense(emsize * 2, hidden_size, weight_init=Uniform(0.1)),
            nn.Dropout(keep_prob=dropout_prob),
            nn.ReLU(),
            nn.Dense(hidden_size, nitem, weight_init=Uniform(0.1)),
            nn.Softmax(axis=1)
        )
        self.treat2_layer = nn.SequentialCell(
            nn.Dense(emsize * 2, hidden_size, weight_init=Uniform(0.1)),
            nn.Dropout(keep_prob=dropout_prob),
            nn.ReLU(),
            nn.Dense(hidden_size, nitem, weight_init=Uniform(0.1)),
            nn.Softmax(axis=1)
        )
        self.treat3_layer = nn.SequentialCell(
            nn.Dense(emsize * 3, hidden_size, weight_init=Uniform(0.1)),
            nn.Dropout(keep_prob=dropout_prob),
            nn.ReLU(),
            nn.Dense(hidden_size, ntoken, weight_init=Uniform(0.1)),
            nn.Softmax(axis=1)
        )

        self.mul = ops.Mul()

    def encoder(self, user, item, sentiment_index, fea):  # hidden0_init
        u_emb = self.user_embeddings(user)  # (batch_size, emsize)
        i_emb = self.item_embeddings(item)
        sentiment_feature = self.sentiment_embeddings(sentiment_index)
        fea_emb = self.word_embeddings(fea).squeeze()
        z2 = self.confounder2_layer(ops.concat([u_emb, i_emb], 1))
        z3 = self.confounder3_layer(ops.concat((u_emb, i_emb, fea_emb), 1))
        unit_emb = ops.concat([u_emb, i_emb, sentiment_feature, z2, z3], 1)
        hidden0 = self.tanh(self.trans_linear(unit_emb))
        return hidden0.expand_dims(0)  # [1, batch_size, rnn_dim]

    def decoder(self, seq, fea, new_state):
        seq_emb = self.word_embeddings(seq)
        fea_emb = self.word_embeddings(fea)
        _, hidden_w = self.gru_w(seq_emb, new_state)
        _, hidden_f = self.gru_f(fea_emb, new_state)
        gamma = self.gfru(hidden_w, hidden_f)
        new_state = (1.0 - gamma) * hidden_w + gamma * hidden_f
        decoded = self.predict_linear(new_state.transpose(1, 0, 2))
        return self.logsoftmax(decoded), new_state

    def construct(self, user, item, sentiment_index, seq, fea):
        hidden = self.encoder(user, item, sentiment_index, fea)
        total_word_prob = None
        for word_id in range(seq[0].size):
            inputs = seq[:, word_id:word_id + 1]
            if word_id == 0:
                log_word_prob, hidden = self.decoder(inputs, fea, hidden)
                total_word_prob = log_word_prob
            else:
                log_word_prob, hidden = self.decoder(inputs, fea, hidden)
                total_word_prob = ops.concat([total_word_prob, log_word_prob], 1)
        return total_word_prob  # (batch_size, seq_len, ntoken)

    def predict_rating(self, user, item):  # (batch_size,)
        user_emb = self.user_embeddings(user)  # (batch_size, emsize)
        item_emb = self.item_embeddings(item)
        z1 = self.confounder1_layer(ops.concat((user_emb, item_emb), 1))
        ui_concat = ops.concat([user_emb, item_emb, z1], 1)  # (batch_size, emsize * 2)
        hidden = self.relu(self.first_layer(ui_concat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.relu(layer(hidden))  # (batch_size, hidden_size)
            hidden = self.dropout(hidden)
        rating = ops.squeeze(self.last_layer(hidden))  # (batch_size,)

        return rating  # (batch_size,)

    def predict_puiz1(self, user, item):
        batch_size = user.size

        user_emb_model = self.user_embeddings(user)
        item_emb_model = self.item_embeddings(item)
        z1 = self.confounder1_layer(ops.concat((user_emb_model, item_emb_model), 1))

        user_emb = self.user_embeddings_mlp(user)  # (batch, emsize)
        user_z1_emb = self.mul(user_emb, z1)
        item_emb_tot = self.item_embeddings_mlp.embedding_table

        all_score_1 = None
        all_score_2 = None

        score = np.zeros(batch_size)
        for index in range(batch_size):
            ui_score = self.mul(user_z1_emb[index], item_emb_tot).sum(1) / self.t
            all_score_1 = ui_score
            ui_score = self.softmax(ui_score)
            all_score_2 = ui_score
            score[index] = ui_score[item[index]]
        return score, all_score_1, all_score_2

    def predict_puiz2(self, user, item):
        batch_size = user.size

        user_emb_model = self.user_embeddings(user)
        item_emb_model = self.item_embeddings(item)
        z2 = self.confounder2_layer(ops.concat((user_emb_model, item_emb_model), 1))

        user_emb = self.user_embeddings_mlp(user)
        user_z2_emb = self.mul(user_emb, z2)
        item_emb_tot = self.item_embeddings_mlp.embedding_table

        score = np.zeros(batch_size)
        for index in range(batch_size):
            ui_score = self.mul(user_z2_emb[index], item_emb_tot).sum(1) / self.t
            ui_score = self.softmax(ui_score)
            score[index] = ui_score[item[index]]
        return score

    def predict_puiz3_f(self, user, item, fea, fea_trans):
        batch_size = user.size

        user_emb_model = self.user_embeddings(user)
        item_emb_model = self.item_embeddings(item)
        fea_emb_model = self.word_embeddings(fea).squeeze()
        z3 = self.confounder3_layer(ops.concat((user_emb_model, item_emb_model, fea_emb_model), 1))

        user_emb = self.user_embeddings_mlp(user)
        item_emb = self.item_embeddings_mlp(item)
        user_item_emb = self.mul(user_emb, item_emb)
        user_item_z3_emb = self.mul(user_item_emb, z3)
        fea_emb_tot = self.feature_embeddings_mlp.embedding_table

        score = np.zeros(batch_size)
        for index in range(batch_size):
            ui_score = self.mul(user_item_z3_emb[index], fea_emb_tot).sum(1) / self.t
            ui_score = self.softmax(ui_score)
            score[index] = ui_score[fea_trans[index]]
        return score

    # treatment Prediction
    def predict_treat1(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        z1 = self.confounder1_layer(ops.concat((user_emb, item_emb), 1))
        treat1 = self.treat1_layer(ops.concat((user_emb, z1), 1))
        return treat1

    def predict_treat2(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        z2 = self.confounder2_layer(ops.concat((user_emb, item_emb), 1))
        treat2 = self.treat2_layer(ops.concat((user_emb, z2), 1))
        return treat2

    def predict_treat3(self, user, item, fea):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        fea_emb = self.word_embeddings(fea).squeeze()
        z3 = self.confounder3_layer(ops.concat((user_emb, item_emb, fea_emb), 1))
        treat3 = self.treat3_layer(ops.concat((user_emb, item_emb, z3), 1))
        return treat3
