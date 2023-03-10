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

import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.nn.transformer import TransformerEncoder
from mindspore.train.serialization import load_param_into_net
from mindspore.common.initializer import initializer, XavierUniform

from src.models.modules import PatchEmbed
from src.models.mae_vit import MAEModule
from src.helper import get_2d_sincos_pos_embed


class Vit(MAEModule):
    """pass"""
    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 channels=3,
                 dropout=0.,
                 drop_path=0.1,
                 global_pool=True,
                 initialization=XavierUniform()):
        super(Vit, self).__init__(batch_size, image_size, patch_size)
        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))
        seq_length = self.num_patches + 1
        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, seq_length, encoder_dim)),
            name='pos_embedding', requires_grad=False
        )

        self.encoder = TransformerEncoder(batch_size=batch_size, num_layers=encoder_layers,
                                          num_heads=encoder_num_heads, hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim*mlp_ratio, seq_length=seq_length,
                                          hidden_dropout_rate=drop_path)

        self.add = P.Add()
        self.cast = P.Cast()
        self.cat = P.Concat(axis=1)
        self.norm = nn.LayerNorm((encoder_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.fc_norm = nn.LayerNorm((encoder_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.global_pool = global_pool
        self.reduce_mean = P.ReduceMean()

        # self.stem = VitStem(encoder_dim, patch_size, image_size)
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,
                                      in_features=channels, out_features=encoder_dim)
        if dropout:
            self.is_dropout = True
            self.dropout = nn.Dropout(p=dropout)
        self.encoder_input_mask = Tensor(np.ones((batch_size, seq_length, seq_length)),
                                         mstype.float32)

        self._init_weights()

    def _init_weights(self):
        encoder_pos_emd = Tensor(
            get_2d_sincos_pos_embed(self.encoder_pos_embedding.shape[-1],
                                    int(self.num_patches ** .5),
                                    cls_token=True),
            mstype.float32
        )
        self.encoder_pos_embedding.set_data(P.ExpandDims()(encoder_pos_emd, 0))

    def construct(self, img):
        tokens = self.patch_embed(img)
        tokens = self.cat((self.cls_token, tokens))
        tokens = self.add(tokens, self.encoder_pos_embedding)

        if self.is_dropout:
            temp = self.cast(tokens, mstype.float32)
            temp = self.dropout(temp)
            tokens = self.cast(temp, tokens.dtype)

        tokens = self.encoder(tokens, self.encoder_input_mask)[0]

        if self.global_pool:
            token = tokens[:, 1:, :]
            tokens = self.reduce_mean(token, 1)
            out = self.fc_norm(tokens)
        else:
            tokens = self.norm(tokens)
            out = tokens[:, 0]

        return out


class FineTuneVit(nn.Cell):
    """Fintune Vit from Mae Model."""
    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 num_classes=1001,
                 dropout=0.,
                 drop_path=0.1,
                 initialization=XavierUniform(),
                 **kwargs):
        super(FineTuneVit, self).__init__()
        self.encoder = Vit(batch_size, patch_size, image_size,
                           dropout=dropout, drop_path=drop_path, **kwargs)
        encoder_dim = kwargs["encoder_dim"]
        self.head = nn.Dense(encoder_dim, num_classes)
        self.head.weight.set_data(initializer(initialization, [num_classes, encoder_dim]))

    def init_weights(self, param_dict):
        """Full model weights initialization."""
        net_not_load, _ = load_param_into_net(self, param_dict)
        return net_not_load

    @staticmethod
    def no_weight_decay():
        return {'encoder.cls_token', 'encoder.encoder_pos_embedding'}

    def construct(self, img):
        tokens = self.encoder(img)
        return self.head(tokens)
