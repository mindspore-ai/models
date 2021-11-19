# Copyright 2021 Huawei Technologies Co., Ltd
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
"""transformer for cct"""
import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

from src.models.cct.misc import DropPath1D, Identity


class Attention(nn.Cell):
    """Attention Block"""

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)
        self.attn_drop = nn.Dropout(keep_prob=1 - attention_dropout)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - projection_dropout)
        self.matmul = ops.BatchMatMul()
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct"""
        B, N, _ = x.shape
        q = ops.Reshape()(self.q(x), (B, N, self.num_heads, -1))
        q = ops.Transpose()(q, (0, 2, 1, 3))

        k = ops.Reshape()(self.k(x), (B, N, self.num_heads, -1))
        k = ops.Transpose()(k, (0, 2, 1, 3))

        v = ops.Reshape()(self.v(x), (B, N, self.num_heads, -1))
        v = ops.Transpose()(v, (0, 2, 1, 3))

        attn = self.matmul(q, ops.Transpose()(k, (0, 1, 3, 2)))
        attn = self.softmax(attn * self.scale)
        attn = self.attn_drop(attn)

        x = ops.Transpose()(self.matmul(attn, v), (0, 2, 1, 3))
        x = ops.Reshape()(x, (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Cell):
    """TransformerEncoderLayer"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.fc1 = nn.Dense(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(keep_prob=1 - dropout)
        self.norm1 = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.fc2 = nn.Dense(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(keep_prob=1 - dropout)

        self.drop_path1 = DropPath1D(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.drop_path2 = DropPath1D(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = nn.GELU()

    def construct(self, src, *args, **kwargs):
        src = src + self.drop_path1(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.fc2(self.dropout1(self.activation(self.fc1(src))))
        src = src + self.drop_path2(self.dropout2(src2))
        return src


class TransformerClassifier(nn.Cell):
    """TransformerClassifier"""

    def __init__(self, seq_pool=True, embedding_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.0, num_classes=1000,
                 dropout=0.1, attention_dropout=0.1, stochastic_depth=0.1, positional_embedding='learnable',
                 sequence_length=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(Tensor(np.zeros(1, 1, self.embedding_dim), mstype.float32), requires_grad=True)
        else:
            self.attention_pool = nn.Dense(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(Tensor(np.zeros([1, sequence_length, embedding_dim]), mstype.float32),
                                                requires_grad=True)
                self.positional_emb.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.2),
                                                                     self.positional_emb.shape))
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        dpr = [x for x in np.linspace(0, stochastic_depth, num_layers)]
        self.blocks = nn.CellList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm((embedding_dim,), epsilon=1e-5)

        self.fc = nn.Dense(embedding_dim, num_classes)
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        """TransformerClassifier construct"""
        if self.positional_emb is None and x.shape[1] < self.sequence_length:
            x = ops.Pad((0, 0), (0, self.n_channels - x.shape[1]))(x)
        if not self.seq_pool:
            cls_token = ops.Tile()(self.class_emb, (x.shape[0], 1, 1))
            x = ops.Concat()((cls_token, x), 1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = ops.BatchMatMul()(ops.Transpose()(self.softmax(self.attention_pool(x)), (0, 2, 1)), x)
            x = ops.Squeeze(-2)(x)
        else:
            x = x[:, 0]
        x = self.fc(x)
        return x

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        """get sinusoidal embedding"""
        pe = np.array([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                       for p in range(n_channels)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)
        return Tensor(pe, mstype.float32)
