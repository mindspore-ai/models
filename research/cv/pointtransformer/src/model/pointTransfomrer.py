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
from mindspore import nn, Tensor, dtype

from src.model.point_helper import TransitionDown, TransitionUp, Dense16
from src.model.pointTransformerBlock import PointTransformerBlock

class PointTransformerCls(nn.Cell):
    def __init__(self, block=None, blocks=None, in_chans=6, num_classes=40, batch_norm=nn.BatchNorm1d):
        super().__init__()
        self.in_planes, planes = in_chans, [32, 64, 128, 256, 512]
        share_planes = 4
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0],
                                   drop=0.8, attn_drop_rate=0.8)  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1],
                                   drop=0.8, attn_drop_rate=0.8)  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2],
                                   drop=0.8, attn_drop_rate=0.8)  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3],
                                   drop=0.8, attn_drop_rate=0.8)  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4],
                                   drop=0.8, attn_drop_rate=0.8)  # N/256

        self.head = nn.SequentialCell(
            Dense16(planes[4], 256, weight_init='he_uniform'),
            batch_norm(256, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.5),
            Dense16(256, 64, weight_init='he_uniform'),
            batch_norm(64, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.5),
            Dense16(64, num_classes, weight_init='he_uniform')
        )
        self._initialize_weights()

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16, drop=0.9, attn_drop_rate=0.9):
        layers = []
        layers.append(TransitionDown(stride, nsample, self.in_planes, planes))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample, drop, attn_drop_rate))
        return nn.SequentialCell(*layers)

    def _initialize_weights(self):
        np.random.seed(0)
        self.init_parameters_data()
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                             cell.weight.data.shape).astype("float32")))
                if cell.bias is not None:
                    cell.bias.set_data(
                        Tensor(np.zeros(cell.bias.data.shape, dtype="float32")))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(
                    Tensor(np.ones(cell.gamma.data.shape, dtype="float32")))
                cell.beta.set_data(
                    Tensor(np.zeros(cell.beta.data.shape, dtype="float32")))
            elif isinstance(cell, nn.BatchNorm1d):
                cell.gamma.set_data(
                    Tensor(np.ones(cell.gamma.data.shape, dtype="float32")))
                cell.beta.set_data(
                    Tensor(np.zeros(cell.beta.data.shape, dtype="float32")))
            elif isinstance(cell, nn.Dense):
                cell.to_float(dtype.float16)
                cell.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, cell.weight.data.shape).astype("float16")))
                if cell.bias is not None:
                    cell.bias.set_data(
                        Tensor(np.zeros(cell.bias.data.shape, dtype="float16")))

    def construct(self, point):
        pos = point[..., :3]
        feats1, pos1 = self.enc1([point, pos])
        feats2, pos2 = self.enc2([feats1, pos1])
        feats3, pos3 = self.enc3([feats2, pos2])
        feats4, pos4 = self.enc4([feats3, pos3])
        feats5, _ = self.enc5([feats4, pos4])

        x = feats5.mean(1)
        x = self.head(x)
        return x

def create_cls_mode():
    return PointTransformerCls(PointTransformerBlock, [2, 3, 3, 3, 2])


class PointTransformerSeg(nn.Cell):
    def __init__(self, block, num_blocks, in_chans=6, num_classes=50):
        super().__init__()
        self.in_planes, dims = in_chans, [32, 64, 128, 256, 512]
        mlp_ratio = 4
        stride, num_neighbors = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, dims[0], num_blocks[0], mlp_ratio, stride[0], num_neighbors[0],
                                   drop=0.7, attn_drop_rate=0.7)  # N/1
        self.enc2 = self._make_enc(block, dims[1], num_blocks[1], mlp_ratio, stride[1], num_neighbors[1],
                                   drop=0.7, attn_drop_rate=0.7)  # N/4
        self.enc3 = self._make_enc(block, dims[2], num_blocks[2], mlp_ratio, stride[2], num_neighbors[2],
                                   drop=0.7, attn_drop_rate=0.7)  # N/16
        self.enc4 = self._make_enc(block, dims[3], num_blocks[3], mlp_ratio, stride[3], num_neighbors[3],
                                   drop=0.6, attn_drop_rate=0.6)  # N/64
        self.enc5 = self._make_enc(block, dims[4], num_blocks[4], mlp_ratio, stride[4], num_neighbors[4],
                                   drop=0.6, attn_drop_rate=0.6)  # N/256
        self.dec5 = self._make_dec(block, dims[4], 2, mlp_ratio, num_neighbors[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, dims[3], 2, mlp_ratio, num_neighbors[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, dims[2], 2, mlp_ratio, num_neighbors[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, dims[1], 2, mlp_ratio, num_neighbors[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, dims[0], 2, mlp_ratio, num_neighbors[0])  # fusion p2 and p1

        self.head = nn.SequentialCell(
            Dense16(dims[0], 512, weight_init='he_uniform'),
            nn.ReLU(),
            nn.Dropout(0.5),
            Dense16(512, 256, weight_init='he_uniform'),
            nn.ReLU(),
            nn.Dropout(0.5),
            Dense16(256, num_classes, weight_init='he_uniform')
        )
        self._initialize_weights()

    def _make_enc(self, block, planes, num_blocks,
                  mlp_ratio=8, stride=1, num_neighbors=16, drop=0.9, attn_drop_rate=0.9):
        layers = []
        layers.append(TransitionDown(stride, num_neighbors, self.in_planes, planes))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, self.in_planes, mlp_ratio, num_neighbors, drop, attn_drop_rate))
        return nn.SequentialCell(*layers)

    def _make_dec(self, block, planes, num_blocks, mlp_ratio=8, num_neighbors=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, self.in_planes, mlp_ratio, num_neighbors, 0.6, 0.6))
        return nn.CellList(layers)

    def _initialize_weights(self):
        np.random.seed(0)
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm1d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.to_float(dtype.float16)
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float16")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float16")))

    def construct(self, point):
        pos = point[..., :3]
        feats1, pos1 = self.enc1([point, pos])
        feats2, pos2 = self.enc2([feats1, pos1])
        feats3, pos3 = self.enc3([feats2, pos2])
        feats4, pos4 = self.enc4([feats3, pos3])
        feats5, pos5 = self.enc5([feats4, pos4])

        x5 = self.dec5[0]([feats5, pos5])
        x5 = self.dec5[1]([x5, pos5])[0]
        x4 = self.dec4[0]([feats4, pos4], [x5, pos5])
        x4 = self.dec4[1]([x4, pos4])[0]
        x3 = self.dec3[0]([feats3, pos3], [x4, pos4])
        x3 = self.dec3[1]([x3, pos3])[0]
        x2 = self.dec2[0]([feats2, pos2], [x3, pos3])
        x2 = self.dec2[1]([x2, pos2])[0]
        x1 = self.dec1[0]([feats1, pos1], [x2, pos2])
        x1 = self.dec1[1]([x1, pos1])[0]

        x = self.head(x1)
        return x

def create_seg_mode():
    return PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3])
