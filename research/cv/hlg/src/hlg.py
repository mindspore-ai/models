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
import math
from importlib import import_module
from easydict import EasyDict as edict
import numpy as np

import mindspore
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

import mindspore.nn as nn
import mindspore.ops as ops

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0
        self.rand = ops.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = ops.Shape()
        self.floor = ops.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x

class HLGConfig:
    """
    HLGConfig
    """
    def __init__(self, configs):
        self.configs = configs

        # network init
        self.network_init = mindspore.common.initializer.Normal(sigma=1.0)
        self.network_dropout_rate = 0.0
        self.network = HLG
        self.pos_dropout_rate = 0.0

        # patch_embed
        self.patch_init = mindspore.common.initializer.HeUniform(math.sqrt(5))
        self.patchembed = PatchEmbed

        # body
        self.body_norm = nn.LayerNorm
        self.body_drop_path_rate = 0.1
        self.body = Transformer

        # body attention
        self.attention_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.attention_activation = nn.Softmax()
        self.attention_dropout_rate = 0.0
        self.project_dropout_rate = 0.0
        self.attention = Attention

        # body feedforward
        self.feedforward_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.feedforward_activation = nn.GELU()
        self.feedforward_dropout_rate = 0.0
        self.feedforward = FeedForward

        # head
        self.head_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.head_activation = nn.GELU()


class ResidualCell(nn.Cell):
    """Cell which implements x + f(x) function."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x, **kwargs):
        return self.cell(x, **kwargs) + x


class PatchEmbed(nn.Cell):
    # Image to Patch Embedding

    def __init__(self, image_size=224, kernel_size=7, in_channels=3, d_model=768, patch_size=16):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size {image_size} should be divided by patch_size {patch_size}."
        self.num_patches = (image_size // patch_size) ** 2
        he_uniform = mindspore.common.initializer.HeUniform(math.sqrt(5))
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2,
                               pad_mode='pad', padding=1, has_bias=True, weight_init=he_uniform)
        self.norm1 = nn.BatchNorm2d(num_features=32, momentum=0.9)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                               pad_mode='pad', padding=1, has_bias=True, weight_init=he_uniform)
        self.norm2 = nn.BatchNorm2d(num_features=32, momentum=0.9)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=d_model, kernel_size=3, stride=2,
                               pad_mode='pad', padding=1, has_bias=True, weight_init=he_uniform)
        self.norm3 = nn.BatchNorm2d(num_features=d_model, momentum=0.9)
        self.gelu = nn.GELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=kernel_size, stride=patch_size,
                              pad_mode='pad', padding=1, has_bias=True, weight_init=he_uniform)
        self.norm = nn.BatchNorm2d(num_features=d_model, momentum=0.9)

    def construct(self, x):
        if self.kernel_size == 7:
            x = self.gelu(self.norm1(self.conv1(x)))
            x = self.gelu(self.norm2(self.conv2(x)))
            x = self.gelu(self.norm3(self.conv3(x)))
        else:
            x = self.gelu(self.norm(self.conv(x)))
        return x


class DynamicPosBias(nn.Cell):
    """Dynamic Position embeding"""
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        truncated_normal = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.pos_proj = nn.Dense(in_channels=2, out_channels=self.pos_dim, weight_init=truncated_normal)
        self.pos1 = nn.SequentialCell([
            nn.LayerNorm(normalized_shape=(self.pos_dim,), epsilon=1e-05),
            nn.ReLU(),
            nn.Dense(in_channels=self.pos_dim, out_channels=self.pos_dim, weight_init=truncated_normal),
        ])
        self.pos2 = nn.SequentialCell([
            nn.LayerNorm(normalized_shape=(self.pos_dim,), epsilon=1e-05),
            nn.ReLU(),
            nn.Dense(in_channels=self.pos_dim, out_channels=self.pos_dim, weight_init=truncated_normal)
        ])
        self.pos3 = nn.SequentialCell([
            nn.LayerNorm(normalized_shape=(self.pos_dim,), epsilon=1e-05),
            nn.ReLU(),
            nn.Dense(in_channels=self.pos_dim, out_channels=self.num_heads, weight_init=truncated_normal)
        ])
    def construct(self, biases):
        # print("In DynamicPosBias, biases.shape={}".format(biases.shape))
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos += self.pos1(pos)
            pos += self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention(nn.Cell):
    """Attention layer implementation."""

    def __init__(self, hlg_config, d_model, dim_head, heads):
        super().__init__()
        initialization = hlg_config.attention_init
        activation = hlg_config.attention_activation
        attn_drop = hlg_config.attention_dropout_rate
        proj_drop = hlg_config.project_dropout_rate

        inner_dim = heads * dim_head
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Dense(d_model, inner_dim, has_bias=True, weight_init=initialization)
        self.to_k = nn.Dense(d_model, inner_dim, has_bias=True, weight_init=initialization)
        self.to_v = nn.Dense(d_model, inner_dim, has_bias=True, weight_init=initialization)

        self.proj = nn.Dense(inner_dim, d_model, has_bias=True, weight_init=initialization)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj_drop = nn.Dropout(1 - proj_drop)
        self.activation = activation

        #auxiliary functions
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = ops.BatchMatMul()
        self.unstack = ops.Unstack(0)
        self.meshgrid = ops.Meshgrid(indexing="ij")
        self.stack = ops.Stack()
        self.flatten = ops.Flatten()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x, H=None, W=None, h0_token=None, group_size=(7, 7)):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model, h, d = x.shape[0], x.shape[1], x.shape[2], self.heads, self.dim_head

        if h0_token is not None:
            _, seq_len_h0, _ = h0_token.shape[0], h0_token.shape[1], h0_token.shape[2]
            q = ops.transpose(ops.reshape(self.to_q(x), (bs, seq_len, h, d)), (0, 2, 1, 3)) # [bs, h, seq_len, d]
            k = ops.transpose(ops.reshape(self.to_k(h0_token), (bs, seq_len_h0, h, d)), (0, 2, 1, 3))
            v = ops.transpose(ops.reshape(self.to_v(h0_token), (bs, seq_len_h0, h, d)), (0, 2, 1, 3)) # [bs, h, seq_len_h0, d]
        else:
            q = ops.transpose(ops.reshape(self.to_q(x), (bs, seq_len, h, d)), (0, 2, 1, 3)) # [bs, h, seq_len, d]
            k = ops.transpose(ops.reshape(self.to_k(x), (bs, seq_len, h, d)), (0, 2, 1, 3))
            v = ops.transpose(ops.reshape(self.to_v(x), (bs, seq_len, h, d)), (0, 2, 1, 3))

        attn = self.q_matmul_k(q, k) * self.scale # [bs, h, seq_len, seq_len] or [bs, h, seq_len, seq_len_h0]
        attn = self.activation(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v) # bs, head, seq_len, dim_head
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (bs, seq_len, d_model))
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Cell):
    """FeedForward layer implementation."""

    def __init__(self, hlg_config, d_model, mlp_ratio):
        super().__init__()

        hidden_dim = int(d_model * mlp_ratio)

        initialization = hlg_config.feedforward_init
        activation = hlg_config.feedforward_activation
        dropout_rate = hlg_config.feedforward_dropout_rate

        self.ff1 = nn.Dense(d_model, hidden_dim, weight_init=initialization)
        self.activation = activation
        self.dropout = nn.Dropout(keep_prob=1.-dropout_rate)
        self.ff2 = nn.Dense(hidden_dim, d_model, weight_init=initialization)

    def construct(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = self.dropout(x)
        return x


# Block: Attention + Mlp
class Transformer(nn.Cell):
    def __init__(self, hlg_config, dim, heads, dim_head, mlp_ratio, drop_path, fea_size, lsda_flag=0, stage_index=1):
        super().__init__()
        self.lsda_flag = lsda_flag
        self.stage_index = stage_index
        self.norm0 = hlg_config.body_norm((dim,))
        self.norm1 = hlg_config.body_norm((dim,))
        self.norm2 = hlg_config.body_norm((dim,))

        self.H, self.W = fea_size

        self.attn = hlg_config.attention(hlg_config, d_model=dim, dim_head=dim_head, heads=heads)
        self.mlp = hlg_config.feedforward(hlg_config, d_model=dim, mlp_ratio=mlp_ratio)
        self.gelu = nn.GELU()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else ops.Identity()

        self.repeat = 1
        he_uniform = mindspore.common.initializer.HeUniform(math.sqrt(5))
        self.spatial_smooth = nn.CellList([nn.SequentialCell(
            [nn.Conv2d(in_channels=dim, out_channels=dim // 4, kernel_size=1,
                       pad_mode='pad', has_bias=True, weight_init=he_uniform),
             nn.BatchNorm2d(num_features=dim // 4, momentum=0.9),
             nn.GELU(),
             nn.Conv2d(in_channels=dim // 4, out_channels=dim // 4, kernel_size=3,
                       pad_mode='pad', padding=1, group=1, has_bias=True, weight_init=he_uniform),
             nn.BatchNorm2d(num_features=dim // 4, momentum=0.9),
             nn.GELU(),
             nn.Conv2d(in_channels=dim // 4, out_channels=dim, kernel_size=1,
                       pad_mode='pad', has_bias=True, weight_init=he_uniform),
             nn.BatchNorm2d(num_features=dim, momentum=0.9)]) for i in range(self.repeat)])
        # auxiliary functions
        self.mean = ops.ReduceMean()
        self.flatten = ops.Flatten()

    def construct(self, x, stage_index):
        # h0_pos: [1, c, h, w]
        bs, N, c = x.shape
        h0_token = None
        if stage_index == 4:
            x += self.drop_path(self.attn(self.norm1(x), self.H, self.W, h0_token))
            x += self.drop_path(self.mlp(self.norm2(x)))
            return x

        if N == self.H * self.W:
            if self.lsda_flag == 0:
                # 0 for SDA
                h1_x = x
                kH, _, _, newH = 7, 7, 0, self.H // 7 # h_conf
                kW, _, _, newW = 7, 7, 0, self.W // 7 # w_conf

                h1_x = ops.reshape(ops.transpose(h1_x, (0, 2, 1)), (bs, c, self.H, self.W))
                for i in range(self.repeat):
                    h1_x += self.spatial_smooth[i](h1_x)
                    h1_x = self.gelu(h1_x)

                h1_x = ops.transpose(h1_x, (0, 2, 3, 1))
                h1_x = ops.transpose(ops.reshape(h1_x, (bs, newH, kH, newW, kW, c)), (0, 1, 3, 2, 4, 5))
                h1_x = ops.reshape(h1_x, (bs * newH * newW, kH * kW, c))
                h1_x += self.drop_path(self.attn(self.norm0(h1_x), group_size=(7, 7)))

                h0_token = self.mean(h1_x, 1)
                h0_token = ops.reshape(h0_token, (bs, newH*newW, c))

                h1_feature = h1_x # b * newH * newW, kH * kW, c

                x = ops.reshape(ops.transpose(ops.reshape(h1_feature, (bs, newH, newW, kH, kW, c,)), (0, 1, 3, 2, 4, 5)), (bs, newH * kH * newW * kW, c)) # b N c

            elif self.lsda_flag == 1:
                # 1 for LDA
                h1_x = x
                kH, _, _, newH = 7, 7, 0, self.H // 7 # h_conf
                kW, _, _, newW = 7, 7, 0, self.W // 7 # w_conf

                h1_x = ops.reshape(ops.transpose(h1_x, (0, 2, 1)), (bs, c, self.H, self.W))
                for i in range(self.repeat):
                    h1_x += self.spatial_smooth[i](h1_x)
                    h1_x = self.gelu(h1_x)

                h1_x = ops.transpose(h1_x, (0, 2, 3, 1)) # b H W c
                h1_x = ops.transpose(ops.reshape(h1_x, (bs, kH, newH, kW, newW, c)), (0, 2, 4, 1, 3, 5))
                h1_x = ops.reshape(h1_x, (bs * newH * newW, kH * kW, c))
                h1_x += self.drop_path(self.attn(self.norm0(h1_x)))

                h0_token = self.mean(h1_x, 1)
                h0_token = ops.reshape(h0_token, (bs, newH*newW, c))

                h1_feature = h1_x  # b * newH * newW, kH * kW, c
                x = ops.reshape(ops.transpose(ops.reshape(h1_feature, (bs, newH, newW, kH, kW, c)),
                                              (0, 3, 1, 4, 2, 5)), (bs, kH * newH * kW * newW, c))

        x += self.drop_path(self.attn(self.norm1(x), self.H, self.W, h0_token, group_size=(self.H, self.H // 7)))
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x


class HLG(nn.Cell):
    """Vision Transformer implementation."""

    def __init__(self, hlg_config):
        super().__init__()

        num_classes = hlg_config.configs.num_classes
        d_model = hlg_config.configs.d_model
        depth = hlg_config.configs.depth
        num_head = hlg_config.configs.num_head
        dim_head = hlg_config.configs.dim_head
        mlp_ratios = hlg_config.configs.mlp_ratios

        initialization = hlg_config.network_init
        head_initialization = hlg_config.head_init

        # patch_embed
        self.patch_embed1 = hlg_config.patchembed(image_size=224, kernel_size=7, in_channels=3,
                                                  d_model=d_model[0], patch_size=4)
        self.patch_embed2 = hlg_config.patchembed(image_size=56, kernel_size=3, in_channels=d_model[0],
                                                  d_model=d_model[1], patch_size=2)
        self.patch_embed3 = hlg_config.patchembed(image_size=28, kernel_size=3, in_channels=d_model[1],
                                                  d_model=d_model[2], patch_size=2)
        self.patch_embed4 = hlg_config.patchembed(image_size=14, kernel_size=3, in_channels=d_model[2],
                                                  d_model=d_model[3], patch_size=2)

        drop_path_rate = hlg_config.body_drop_path_rate
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depth))]

        cur = 0
        for i in range(depth[0]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            block1 = hlg_config.body(hlg_config, dim=d_model[0], heads=num_head[0], dim_head=dim_head[0],
                                     mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i],
                                     fea_size=(56, 56), lsda_flag=lsda_flag)
            if i == 0:
                self.block1_0 = block1
            elif i == 1:
                self.block1_1 = block1

        cur += depth[0]
        for i in range(depth[1]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            block2 = hlg_config.body(hlg_config, dim=d_model[1], heads=num_head[1], dim_head=dim_head[1],
                                     mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i],
                                     fea_size=(28, 28), lsda_flag=lsda_flag)
            if i == 0:
                self.block2_0 = block2
            elif i == 1:
                self.block2_1 = block2

        cur += depth[1]
        for i in range(depth[2]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            block3 = hlg_config.body(hlg_config, dim=d_model[2], heads=num_head[2], dim_head=dim_head[2],
                                     mlp_ratio=mlp_ratios[2], drop_path=dpr[cur + i],
                                     fea_size=(14, 14), lsda_flag=lsda_flag)
            if i == 0:
                self.block3_0 = block3
            elif i == 1:
                self.block3_1 = block3

        cur += depth[2]
        for i in range(depth[3]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            block4 = hlg_config.body(hlg_config, dim=d_model[3], heads=num_head[3], dim_head=dim_head[3],
                                     mlp_ratio=mlp_ratios[3], drop_path=dpr[cur + i],
                                     fea_size=(7, 7), lsda_flag=lsda_flag)
            if i == 0:
                self.block4_0 = block4
            elif i == 1:
                self.block4_1 = block4

        self.cls_token = Parameter(initializer(initialization, (1, 1, d_model[3])),
                                   name='cls', requires_grad=True)
        '''Position embeding'''
        self.pos_embed1 = Parameter(initializer(initialization, (1, 3136, d_model[0])))
        self.pos_embed2 = Parameter(initializer(initialization, (1, 784, d_model[1])))
        self.pos_embed3 = Parameter(initializer(initialization, (1, 196, d_model[2])))
        self.pos_embed4 = Parameter(initializer(initialization, (1, 49+1, d_model[3])))

        self.concat = ops.Concat(1)
        self.norm = mindspore.nn.LayerNorm((d_model[3],))

        # classification head
        self.head = nn.Dense(in_channels=d_model[3], out_channels=num_classes,
                             weight_init=head_initialization) if num_classes > 0 else ops.Identity()

    def construct(self, img):
        x = self.patch_embed1(img)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        x = x + self.pos_embed1
        x = self.block1_0(x, stage_index=1)
        x = self.block1_1(x, stage_index=1)
        x = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))

        x = self.patch_embed2(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        x = x + self.pos_embed2
        x = self.block2_0(x, stage_index=2)
        x = self.block2_1(x, stage_index=2)
        x = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))

        x = self.patch_embed3(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        x = x + self.pos_embed3
        x = self.block3_0(x, stage_index=3)
        x = self.block3_1(x, stage_index=3)
        x = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))

        x = self.patch_embed4(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        cls_tokens = ops.BroadcastTo((B, -1, -1))(self.cls_token)
        x = self.concat((cls_tokens, x)) # now x has shape = (bs, seq_len+1, d)
        x = x + self.pos_embed4
        x = self.block4_0(x, stage_index=4)
        x = self.block4_1(x, stage_index=4)

        x = self.norm(x)
        x = x[:, 0]

        x = self.head(x)

        return x


def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]
        module = import_module(module_path)
        return getattr(module, name)
    return func_name


hlg_cfg = edict({
    'd_model': (32, 64, 192, 384),
    'depth': (2, 2, 2, 2),
    'heads': (1, 1, 3, 6),
    'mlp_ratios': (8, 8, 4, 4),
    'dim_head': (32, 64, 64, 64),
    'patch_size': 4,
    'image_size': 224,
    'num_classes': 1001,
})

def hlg_mobile(args):
    """hlg_mobile"""
    hlg_cfg.d_model = (32, 64, 192, 384)
    hlg_cfg.depth = (2, 2, 2, 2)
    hlg_cfg.num_head = (2, 4, 8, 16)
    hlg_cfg.mlp_ratios = (8, 8, 4, 4)
    hlg_cfg.dim_head = (16, 16, 24, 24)
    hlg_cfg.patch_size = 4

    hlg_cfg.normalized_shape = hlg_cfg.d_model
    hlg_cfg.image_size = args.train_image_size
    hlg_cfg.num_classes = args.class_num

    if args.hlg_config_path != '':
        print("get hlg_config_path")
        hlg_config = load_function(args.hlg_config_path)(hlg_cfg)
    else:
        print("get default_hlg_cfg")
        hlg_config = HLGConfig(hlg_cfg)

    model = hlg_config.network(hlg_config)

    return model


def get_network(backbone_name, args):
    """get_network"""
    if backbone_name == 'hlg_mobile':
        backbone = hlg_mobile(args=args)
    else:
        raise NotImplementedError
    return backbone
