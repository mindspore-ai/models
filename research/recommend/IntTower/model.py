# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import Reshape, ExpandDims, Transpose, matmul, concat
from mindspore.nn import Embedding
from module import LightSE
import model_config as cfg

class IntTower(nn.Cell):
    """
    IntTower Model Structure
    """

    def __init__(self):
        super(IntTower, self).__init__()
        self.mlp_layers = cfg.mlp_layers
        self.use_multi_layer = cfg.use_multi_layer
        self.activation = 'relu'
        self.head_num = cfg.head_num
        self.feblock_size = cfg.feblock_size
        self.user_embedding_dim = cfg.user_embedding_dim

        self.item_embedding_dim = cfg.item_embedding_dim
        self.sparse_embedding_dim = cfg.sparse_embedding_dim
        self.dropout = nn.Dropout(cfg.keep_rate)
        self.User_SE = LightSE(cfg.user_sparse_field)
        self.user_fe_embedding = None
        self.item_fe_embedding = None

        self.user_bn_list = []
        self.item_bn_list = []

        self.user_dense_layer_1 = nn.Dense(self.user_embedding_dim, self.mlp_layers[0], weight_init='normal',
                                           activation=self.activation)
        self.user_dense_layer_2 = nn.Dense(self.mlp_layers[0], self.mlp_layers[1], weight_init='normal',
                                           activation=self.activation)
        self.user_dense_layer_3 = nn.Dense(self.mlp_layers[1], self.mlp_layers[2], weight_init='normal',
                                           activation=self.activation)

        self.user_fe_layer_1 = nn.Dense(self.mlp_layers[0], self.feblock_size, weight_init='normal',
                                        activation=self.activation)
        self.user_fe_layer_2 = nn.Dense(self.mlp_layers[1], self.feblock_size, weight_init='normal',
                                        activation=self.activation)
        self.user_fe_layer_3 = nn.Dense(self.mlp_layers[2], self.feblock_size, weight_init='normal',
                                        activation=self.activation)

        self.item_dense_layer_1 = nn.Dense(self.item_embedding_dim, self.mlp_layers[0], weight_init='normal',
                                           activation=self.activation)
        self.item_dense_layer_2 = nn.Dense(self.mlp_layers[0], self.mlp_layers[1], weight_init='normal',
                                           activation=self.activation)
        self.item_dense_layer_3 = nn.Dense(self.mlp_layers[1], self.mlp_layers[2], weight_init='normal',
                                           activation=self.activation)

        self.user_bn_layer_1 = nn.BatchNorm1d(self.mlp_layers[0])
        self.user_bn_layer_2 = nn.BatchNorm1d(self.mlp_layers[1])
        self.user_bn_layer_3 = nn.BatchNorm1d(self.mlp_layers[2])

        self.item_bn_layer_1 = nn.BatchNorm1d(self.mlp_layers[0])
        self.item_bn_layer_2 = nn.BatchNorm1d(self.mlp_layers[1])
        self.item_bn_layer_3 = nn.BatchNorm1d(self.mlp_layers[2])

        self.user_dense_list = [self.user_dense_layer_1, self.user_dense_layer_2, self.user_dense_layer_3]
        self.item_dense_list = [self.item_dense_layer_1, self.item_dense_layer_2, self.item_dense_layer_3]
        self.user_fe_list = [self.user_fe_layer_1, self.user_fe_layer_2, self.user_fe_layer_3]

        self.user_bn_list = [self.user_bn_layer_1, self.user_bn_layer_2, self.user_bn_layer_3]
        self.item_bn_list = [self.item_bn_layer_1, self.item_bn_layer_2, self.item_bn_layer_3]

        self.user_fe_dense = nn.Dense(self.mlp_layers[-1], self.feblock_size)
        self.item_fe_dense = nn.Dense(self.mlp_layers[-1], self.feblock_size)
        self.user_id_embedding = Embedding(6040, self.sparse_embedding_dim)
        self.gender_embedding = Embedding(2, self.sparse_embedding_dim)
        self.age_embedding = Embedding(7, self.sparse_embedding_dim)
        self.occupation_embedding = Embedding(21, self.sparse_embedding_dim)
        self.item_id_embedding = Embedding(3668, self.sparse_embedding_dim)
        self.user_bacth_norm = nn.BatchNorm1d(self.user_embedding_dim)
        self.item_bacth_norm = nn.BatchNorm1d(self.item_embedding_dim)

    def construct(self, inputs):
        user_list, user_dense_list = [], []
        item_list, item_dense_list = [], []
        user_id = ms.Tensor(inputs[:, 0:1], dtype=ms.int32)
        user_list.append(self.user_id_embedding(user_id))
        user_gender = ms.Tensor(inputs[:, 2:3], dtype=ms.int32)
        user_list.append(self.gender_embedding(user_gender))
        user_age = ms.Tensor(inputs[:, 3:4], dtype=ms.int32)
        user_list.append(self.age_embedding(user_age))
        user_occ = ms.Tensor(inputs[:, 4:5], dtype=ms.int32)
        user_list.append(self.occupation_embedding(user_occ))

        item_id = ms.Tensor(inputs[:, 1:2], dtype=ms.int32)
        item_list.append(self.item_id_embedding(item_id))

        user_mean_rating = ms.Tensor(inputs[:, 5:6], dtype=ms.float32)
        item_mean_rating = ms.Tensor(inputs[:, 6:7], dtype=ms.float32)

        user_dense_list.append(user_mean_rating)
        item_dense_list.append(item_mean_rating)

        user_sparse_embedding = concat(user_list, axis=1)
        user_sparse_embedding = self.User_SE(user_sparse_embedding)

        user_sparse_input = ops.flatten(user_sparse_embedding)

        item_sparse_embedding = concat(item_list)
        item_sparse_input = ops.flatten(item_sparse_embedding)

        user_dense_input = concat(user_dense_list)
        item_dense_input = concat(item_dense_list)

        user_input = concat([user_sparse_input, user_dense_input], axis=-1)
        item_input = concat([item_sparse_input, item_dense_input], axis=-1)

        user_embed = self.user_bacth_norm(user_input)
        item_embed = self.item_bacth_norm(item_input)
        user_fe_reps = []

        for i in range(len(self.mlp_layers)):
            user_embed = self.dropout(self.user_bn_list[i](
                self.user_dense_list[i](user_embed)))
            if self.use_multi_layer:
                user_fe_rep = self.user_fe_list[i](user_embed)
                user_fe_reps.append(user_fe_rep)

            item_embed = self.dropout(self.item_bn_list[i](
                self.item_dense_list[i](item_embed)))
        item_fe_rep = self.item_fe_dense(item_embed)

        if self.use_multi_layer:
            score = []
            for i in range(len(user_fe_reps)):
                user_temp = Reshape()(user_fe_reps[i],
                                      (-1, self.head_num, int(user_fe_reps[i].shape[1] // self.head_num)))
                item_temp = Reshape()(item_fe_rep, (-1, self.head_num, int(item_fe_rep.shape[1] // self.head_num)))
                item_temp = Transpose()((item_temp), (0, 2, 1))
                dot_col = matmul(user_temp, item_temp)
                max_col = dot_col.max(axis=2)
                sum_col = max_col.sum(axis=1)
                expand_col = ExpandDims()(sum_col, 1)
                score.append(expand_col)
            model_output = concat(score, axis=1).sum(axis=1)
            model_output = nn.Sigmoid()(Reshape()(model_output, (-1, 1)))
        else:
            user_fe_rep = self.user_fe_dense(user_embed)
            user_temp = Reshape()(user_fe_rep, (-1, self.head_num, int(user_fe_rep.shape[1] // self.head_num)))
            item_temp = Reshape()(item_fe_rep, (-1, self.head_num, int(item_fe_rep.shape[1] // self.head_num)))
            item_temp = Transpose()((item_temp), (0, 2, 1))
            dot_col = matmul(user_temp, item_temp)
            max_col = dot_col.max(axis=2)
            sum_col = max_col.sum(axis=1)
            expand_col = ExpandDims()(sum_col, 1)
            score = expand_col
            model_output = nn.Sigmoid()(score)

        return model_output
