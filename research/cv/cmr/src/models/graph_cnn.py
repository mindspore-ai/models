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

from __future__ import division

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import numpy as np

from src.models.graph_layers import GraphResBlock, GraphLinear
from src.models.resnet import resnet50


class GraphCNN(nn.Cell):

    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
        """
        :param A: mesh.adjmat, shape = (1712, 1723)
        :param ref_vertices: mesh.ref_vertices, shape = (1723, 3)
        """
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.resnet = resnet50(pretrained=True)
        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for _ in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.SequentialCell([
            GraphResBlock(num_channels, 64, A),
            GraphResBlock(64, 32, A),
            nn.GroupNorm(32 // 8, 32),
            nn.ReLU(),
            GraphLinear(32, 3)
        ])
        self.gc = nn.SequentialCell(*layers)
        self.camera_fc = nn.SequentialCell([
            nn.GroupNorm(num_channels // 8, num_channels),
            nn.ReLU(),
            GraphLinear(num_channels, 1),
            nn.ReLU(),
            nn.Dense(A.shape[0], 3)
        ])

        self.concat = P.Concat(axis=1)

    def construct(self, image):
        """Forward pass
        Inputs:
            image: shape = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: shape = (B, 3)
        """
        batch_size = image.shape[0]
        ref_vertices = self.ref_vertices
        # ref_vertices shape: (16, 3, 1723)
        ref_vertices = np.broadcast_to(ref_vertices, (batch_size, ref_vertices.shape[0], ref_vertices.shape[1]))
        image_resnet = self.resnet(image)
        image_enc = image_resnet.view(batch_size, 2048, 1)
        # image_enc shape: (16, 2048, 3)
        image_enc = np.broadcast_to(image_enc, (image_enc.shape[0], image_enc.shape[1], ref_vertices.shape[-1]))
        # x shape: (16, 2051, 1723)
        x = self.concat((ref_vertices, image_enc))
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)
        return shape, camera
