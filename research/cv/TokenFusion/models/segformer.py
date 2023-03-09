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

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.ops import stop_gradient
import cfg
from . import mix_transformer


class MLP(nn.Cell):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Dense(input_dim, embed_dim)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        B, C, _, _ = x.shape
        x = self.transpose(self.reshape(x, (B, C, -1)), (0, 2, 1))
        x = self.proj(x)
        return x


class SegFormerHead(nn.Cell):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
            self,
            feature_strides=None,
            in_channels=128,
            embedding_dim=256,
            num_classes=20,
            **kwargs
        ):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout(p=0.9)

        self.linear_fuse_conv = nn.Conv2d(
            embedding_dim * 4, embedding_dim, kernel_size=1
        )
        self.linear_fuse_bn = nn.BatchNorm2d(
            embedding_dim
        )
        self.relu = nn.ReLU()

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1, has_bias=True
        )
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, _, _ = c4.shape
        _c4 = self.transpose(self.linear_c4(c4), (0, 2, 1)).reshape(
            n, -1, c4.shape[2], c4.shape[3]
        )
        _c4 = ops.interpolate(_c4, sizes=c1.shape[2:], coordinate_transformation_mode='half_pixel', mode='bilinear')

        _c3 = self.transpose(self.linear_c3(c3), (0, 2, 1)).reshape(
            n, -1, c3.shape[2], c3.shape[3]
        )
        _c3 = ops.interpolate(_c3, sizes=c1.shape[2:], coordinate_transformation_mode='half_pixel', mode='bilinear')

        _c2 = self.transpose(self.linear_c2(c2), (0, 2, 1)).reshape(
            n, -1, c2.shape[2], c2.shape[3]
        )
        _c2 = ops.interpolate(_c2, sizes=c1.shape[2:], coordinate_transformation_mode='half_pixel', mode='bilinear')

        _c1 = self.transpose(self.linear_c1(c1), (0, 2, 1)).reshape(
            n, -1, c1.shape[2], c1.shape[3]
        )

        _c = self.linear_fuse_conv(self.concat((_c4, _c3, _c2, _c1)))
        _c = self.linear_fuse_bn(_c)
        _c = self.relu(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class WeTr(nn.Cell):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]

        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims

        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
        )

        self.alpha = Parameter(ops.Ones()(cfg.num_parallel, mindspore.float32))
        self.softmax = ops.Softmax(axis=0)

    def construct(self, x):
        _x, masks = self.encoder(x)
        x = [self.decoder(_x[0]), self.decoder(_x[1])]
        ens = 0
        num_parallel = 2
        alpha_soft = self.softmax(self.alpha)
        for l in range(int(num_parallel)):
            ens += alpha_soft[l] * stop_gradient(x[l])
        x.append(ens)
        return tuple(x), masks
