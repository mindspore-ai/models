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
"""
model.
"""
import copy

import mindspore
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.ops.operations as P


def swish(x):
    return x * P.Sigmoid()(x)


class Attention(nn.Cell):
    """Attention"""
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer_num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.attention_head_size2 = Tensor(config.hidden_size / self.num_attention_heads, mindspore.float32)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.out = nn.Dense(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer_attention_dropout_rate)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        """transpose_for_scores"""
        new_x_shape = P.Shape()(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = P.Reshape()(x, new_x_shape)
        return P.Transpose()(x, (0, 2, 1, 3,))

    def construct(self, hidden_states):
        """construct"""
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = mindspore.ops.matmul(query_layer, P.Transpose()(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores / P.Sqrt()(self.attention_head_size2)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = mindspore.ops.matmul(attention_probs, value_layer)
        context_layer = P.Transpose()(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = P.Shape()(context_layer)[:-2] + (self.all_head_size,)
        context_layer = P.Reshape()(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Cell):
    """Mlp"""
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Dense(config.hidden_size, config.transformer_mlp_dim,
                            weight_init='XavierUniform', bias_init='Normal')
        self.fc2 = nn.Dense(config.transformer_mlp_dim, config.hidden_size,
                            weight_init='XavierUniform', bias_init='Normal')
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer_dropout_rate)

    def construct(self, x):
        """construct"""
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None

        if config.patches_grid is not None:
            grid_size = config.patches_grid
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = config.patches_size
            n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size, has_bias=True)
        self.position_embeddings = Parameter(P.Zeros()((1, n_patches+1, config.hidden_size), mindspore.float32),
                                             name="q1", requires_grad=True)
        self.cls_token = Parameter(P.Zeros()((1, 1, config.hidden_size), mindspore.float32), name="q2",
                                   requires_grad=True)

        self.dropout = nn.Dropout(config.transformer_dropout_rate)

    def construct(self, x):
        """construct"""
        B = x.shape[0]
        cls_tokens = P.BroadcastTo((B, self.cls_token.shape[1], self.cls_token.shape[2]))(self.cls_token)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = P.Transpose()(x, (0, 2, 1))
        x = P.Concat(1)((cls_tokens, x))

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Cell):
    """Block"""
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)
        self.ffn_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def construct(self, x):
        """construct"""
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Cell):
    """Encoder"""
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.CellList([])
        self.encoder_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-6)
        for _ in range(config.transformer_num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def construct(self, hidden_states):
        """construct"""
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Cell):
    """Transformer"""
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def construct(self, input_ids):
        """construct"""
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded


class VisionTransformer(nn.Cell):
    """VisionTransformer"""
    def __init__(self, config, img_size=(224, 224), num_classes=21843):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size)
        self.head = nn.Dense(config.hidden_size, num_classes)

    def construct(self, x, labels=None):
        """construct"""
        x = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits
