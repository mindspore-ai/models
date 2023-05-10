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


class MfUi(nn.Cell):
    def __init__(self, nuser, nitem, emsize):
        super(MfUi, self).__init__()
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.softmax = nn.Softmax(axis=0)

    def construct(self, user, item):
        batch_size = user.size
        user_emb = self.user_embeddings(user)
        item_emb_tot = self.item_embeddings.embedding_table
        score = np.zeros(batch_size)
        for index in range(batch_size):
            u_score = ops.mul(user_emb[index], item_emb_tot).sum(1)
            u_score = self.softmax(u_score)
            score[index] = u_score[item[index]]
        return score
