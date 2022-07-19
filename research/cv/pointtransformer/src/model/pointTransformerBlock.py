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

import mindspore.nn as nn
import mindspore.ops as ops

from src.model.point_helper import batched_index_select, Dense16
from src.utils.common import exists

# classes
class PointTransformerLayer(nn.Cell):
    def __init__(self,
                 in_planes,
                 out_planes,
                 share_planes,
                 num_neighbors=None,
                 pos_drop_rate=0.9,
                 attn_drop_rate=0.9,
                 batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.mid_planes = in_planes * share_planes
        self.out_planes = out_planes

        self.num_neighbors = num_neighbors
        self.to_qkv = Dense16(in_planes, in_planes * 3, has_bias=False, weight_init='he_uniform')

        self.func_split = ops.Split(-1, 3)
        self.func_sort = ops.Sort(axis=-1)
        self.func_norm = nn.Norm(axis=-1)
        self.func_softmax = ops.Softmax(axis=-2)
        self.func_sum = ops.ReduceSum()
        self.pos_embed = nn.SequentialCell([Dense16(3, 3 * share_planes, weight_init='he_uniform'),
                                            nn.ReLU(),
                                            Dense16(3 * share_planes, in_planes, weight_init='he_uniform')])
        self.pos_norm = batch_norm(in_planes, momentum=0.1)
        self.pos_drop = nn.Dropout(pos_drop_rate)

        self.attn_mlp = nn.SequentialCell([Dense16(in_planes, self.mid_planes, weight_init='he_uniform'),
                                           nn.ReLU(),
                                           Dense16(self.mid_planes, out_planes, weight_init='he_uniform')])
        self.attn_norm = batch_norm(out_planes, momentum=0.1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def construct(self, x, pos):
        n, num_neighbors = x.shape[1], self.num_neighbors
        # get queries, keys, values
        q, k, v = self.func_split(self.to_qkv(x))

        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]

        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = self.func_norm(rel_pos)
            indices = self.func_sort(rel_dist)[1][:, :, :num_neighbors]
            k = batched_index_select(k, indices)
            v = batched_index_select(v, indices)
            pos_knn = batched_index_select(pos, indices)
            pos_emb = pos[:, :, None, :] - pos_knn
            qk_rel = q[:, :, None, :] - k
        else:
            v = v[:, :, None, :]
            pos_emb = rel_pos
            qk_rel = q[:, :, None, :] - k[:, None, :, :]

        pos_emb = self.pos_norm(self.pos_embed(pos_emb).transpose(0, 3, 2, 1)).transpose(0, 3, 2, 1)
        pos_emb = self.pos_drop(pos_emb)

        v = v + pos_emb
        attn = self.attn_mlp(qk_rel + pos_emb)
        attn = self.attn_norm(attn.transpose(0, 3, 2, 1)).transpose(0, 3, 2, 1)
        attn = self.attn_drop(attn)
        attn = self.func_softmax(attn) #attn -> [B, N, K, C]

        # aggregate
        agg = self.func_sum(attn * v, 2) #sim -> [B, N, C]
        return agg

class PointTransformerBlock(nn.Cell):
    expansion = 1
    def __init__(self,
                 in_planes,
                 planes,
                 share_planes=8,
                 num_neighbors=16,
                 pos_drop_rate=0.9,
                 attn_drop_rate=0.9,
                 batch_norm=nn.BatchNorm1d):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = Dense16(in_planes, planes, has_bias=False, weight_init='he_uniform')
        self.bn1 = batch_norm(planes, momentum=0.1)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, num_neighbors,
                                                 pos_drop_rate, attn_drop_rate)
        self.bn2 = batch_norm(planes, momentum=0.1)
        self.linear2 = nn.Dense(planes, planes * self.expansion, has_bias=False, weight_init='he_uniform')
        self.bn3 = batch_norm(planes * self.expansion, momentum=0.1)
        self.activate_layer = nn.ReLU()
    def construct(self, feats_pos):
        feats, pos = feats_pos
        identity = feats

        B, N, C = feats.shape
        activate_layer = self.activate_layer

        feats = activate_layer(self.bn1(self.linear1(feats).view(-1, C)).view(B, N, C))
        feats = activate_layer(self.bn2(self.transformer(feats, pos).view(-1, C)).view(B, N, C))
        feats = self.bn3(self.linear2(feats).view(-1, C)).view(B, N, C)
        feats += identity
        feats = activate_layer(feats)
        return  [feats, pos]
