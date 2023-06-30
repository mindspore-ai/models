# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore import nn, Tensor

from src.model_utils.common import VERSION_GT_2_0_0


class MLP(nn.Cell):
    def __init__(self, dim, embed_dim):
        super(MLP, self).__init__()
        self.proj = nn.Dense(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x.transpose((0, 2, 1))
        x = self.proj(x)
        return x

    def construct(self, x):
        return self.forward(x)


class ConvModule(nn.Cell):
    def __init__(self, c1, c2, sync_bn=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, has_bias=False, pad_mode='pad')
        if sync_bn:
            self.bn = nn.SyncBatchNorm(c2)
        else:
            self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

    def construct(self, x):
        return self.forward(x)


class SegFormerHead(nn.Cell):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19, sync_bn=False):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims
        self.linear_c4 = MLP(dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(dim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, sync_bn)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1, has_bias=True)

        self.dropout = nn.Dropout2d(0.1)
        self.version_gt_2_0_0 = VERSION_GT_2_0_0

    def construct(self, x):
        c1, c2, c3, c4 = x
        n, _, h, w = c1.shape
        _c4 = self.linear_c4(c4).transpose(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        if self.version_gt_2_0_0:
            _c4 = ops.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)
        else:
            _c4 = ops.interpolate(_c4, sizes=(h, w), mode='bilinear', coordinate_transformation_mode='half_pixel')

        _c3 = self.linear_c3(c3).transpose(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        if self.version_gt_2_0_0:
            _c3 = ops.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)
        else:
            _c3 = ops.interpolate(_c3, sizes=(h, w), mode='bilinear', coordinate_transformation_mode='half_pixel')

        _c2 = self.linear_c2(c2).transpose(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        if self.version_gt_2_0_0:
            _c2 = ops.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)
        else:
            _c2 = ops.interpolate(_c2, sizes=(h, w), mode='bilinear', coordinate_transformation_mode='half_pixel')

        _c1 = self.linear_c1(c1).transpose(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = ops.Concat(axis=1)([_c4, _c3, _c2, _c1])

        seg = self.linear_fuse(_c)
        seg = self.dropout(seg)
        seg = self.linear_pred(seg)
        return seg
