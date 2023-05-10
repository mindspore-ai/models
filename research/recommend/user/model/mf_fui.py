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

import mindspore.nn as nn
import mindspore.numpy as np
from mindspore import ops


class MfFui(nn.Cell):
    def __init__(self, nuser, nitem, nfeature, emsize):
        super(MfFui, self).__init__()

        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.feature_embeddings = nn.Embedding(nfeature, emsize)
        self.softmax = nn.Softmax(axis=0)

    def construct(self, user, item, fea_trans):
        batch_size = user.size
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        user_item_emb = ops.mul(user_emb, item_emb)
        fea_emb_tot = self.feature_embeddings.embedding_table
        score = np.zeros(batch_size)
        for index in range(batch_size):
            ui_score = ops.mul(user_item_emb[index], fea_emb_tot).sum(1)
            ui_score = self.softmax(ui_score)
            score[index] = ui_score[fea_trans[index]]
        return score
