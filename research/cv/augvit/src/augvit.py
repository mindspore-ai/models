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
"""Augvit"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter

class MLP(nn.Cell):
    """MLP"""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dropout = nn.Dropout(1. - dropout)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.act = nn.GELU()

    def construct(self, x):
        """MLP"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Cell):
    """Multi-head Attention"""

    def __init__(self, dim, hidden_dim=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, hidden_dim * 3, has_bias=qkv_bias)
        self.softmax = nn.Softmax(axis=-1)
        self.batmatmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.attn_drop = nn.Dropout(1. - attn_drop)
        self.batmatmul = P.BatchMatMul()
        self.proj = nn.Dense(hidden_dim, dim)
        self.proj_drop = nn.Dropout(1. - proj_drop)

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        """Multi-head Attention"""
        B, N, _ = x.shape
        qkv = self.transpose(self.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.softmax(self.batmatmul_trans_b(q, k) * self.scale)
        attn = self.attn_drop(attn)
        x = self.reshape(self.transpose(self.batmatmul(attn, v), (0, 2, 1, 3)), (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropConnect(nn.Cell):
    """drop connect implementation"""

    def __init__(self, drop_connect_rate=0., seed=0):
        super(DropConnect, self).__init__()
        self.keep_prob = 1 - drop_connect_rate
        seed = min(seed, 0) # always be 0
        self.rand = P.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        """drop connect implementation"""
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


class Block(nn.Cell):
    """augvit Block"""

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, dropout=0., attn_dropout=0., drop_connect=0.):
        super(Block).__init__()
        # transformer
        self.norm1 = nn.LayerNorm([dim])
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_dropout,
                              proj_drop=dropout)
        self.drop_connect = DropConnect(drop_connect)
        self.norm2 = nn.LayerNorm([dim])
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        # aug path
        self.augs_attn = nn.Dense(dim, dim, has_bias=True)
        self.augs = nn.Dense(dim, dim, has_bias=True)
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """augvit Block"""
        x_norm1 = self.norm1(x)
        x = x + self.drop_connect(self.attn(x_norm1)) + self.augs_attn(x_norm1)
        x_norm2 = self.norm2(x)
        x = x + self.drop_connect(self.mlp(x_norm2)) + self.augs(x_norm2)
        return x


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, img_size, patch_size=16, in_channels=3, embedding_dim=768):
        super(PatchEmbed, self).__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """Image to Patch Embedding"""
        x = self.proj(x) # B, 768, 14, 14
        B, C, H, W = x.shape
        x = self.reshape(x, (B, C, H * W))
        x = self.transpose(x, (0, 2, 1)) # B, N, C
        return x


class AugVIT(nn.Cell):
    """augvit"""

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            embedding_dim=768,
            num_heads=12,
            depth=12,
            mlp_ratio=4,
            qkv_bias=False,
            num_class=1000,
            stride=4,
            dropout=0,
            attn_dropout=0,
            drop_path_rate=0
    ):
        super(AugVIT, self).__init__()
        assert embedding_dim % num_heads == 0
        assert img_size % patch_size == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = int((img_size // patch_size) ** 2)
        self.stride = stride
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size, in_channels=in_channels, embedding_dim=embedding_dim)
        self.cls_token = Parameter(Tensor(np.random.rand(1, 1, embedding_dim),
                                          mstype.float32), requires_grad=True)
        self.pos_embed = Parameter(Tensor(np.zeros((1, self.num_patches + 1, embedding_dim)),
                                          mstype.float32), name='pos_embed', requires_grad=True)
        self.pos_drop = nn.Dropout(1. - dropout)
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        layers = []
        for i in range(depth):
            layers.append(Block(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                dropout=dropout, attn_dropout=attn_dropout, drop_connect=dpr[i]))
        self.blocks = nn.SequentialCell(layers)
        self.norm = nn.LayerNorm([embedding_dim])
        self.head = nn.Dense(embedding_dim, num_class)
        self.concat = P.Concat(axis=1)
        self.tile = P.Tile()

    def forward_features(self, x):
        """augvit"""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.tile(self.cls_token, (B, 1, 1))
        x = self.concat((cls_token, x))
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def construct(self, x):
        """augvit"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def augvit_s(num_class):
    """augvit_s"""
    return AugVIT(patch_size=16, embedding_dim=384,
                  depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_class=num_class)
