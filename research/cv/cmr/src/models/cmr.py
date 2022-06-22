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

from mindspore import nn, ops
from mindspore.ops import operations as P
from mindspore import numpy as np

from src.models.graph_cnn import GraphCNN
from src.models.smpl_param_regressor import SMPLParamRegressor


class CMR(nn.Cell):
    """
    A CMR class consists of graph_cnn, smpl_param_regressor, smpl and mesh, from which smpl and mesh are un-trainable
    """
    def __init__(self, mesh, smpl, num_layers, num_channels):
        super(CMR, self).__init__()
        self.graph_cnn = GraphCNN(mesh.adjmat, mesh.ref_vertices.transpose(),
                                  num_layers, num_channels)
        self.smpl_param_regressor = SMPLParamRegressor()
        self.smpl = smpl
        self.mesh = mesh

        self.concat = P.Concat(axis=-1)

    def construct(self, image):
        """Fused forward pass for the 2 networks
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed non-parametric shape: size = (B, 6890, 3)
            Regressed SMPL shape: size = (B, 6890, 3)
            Weak-perspective camera: size = (B, 3)
            SMPL pose parameters (as rotation matrices): size = (B, 24, 3, 3)
            SMPL shape parameters: size = (B, 10)
        """
        batch_size = image.shape[0]
        pred_vertices_sub, camera = self.graph_cnn(image)
        pred_vertices = self.mesh(pred_vertices_sub.transpose((0, 2, 1)))

        x = pred_vertices_sub.transpose((0, 2, 1))
        x = ops.stop_gradient(x)
        x = self.concat((x, np.broadcast_to(self.mesh.ref_vertices,
                                            (batch_size, self.mesh.ref_vertices.shape[0],
                                             self.mesh.ref_vertices.shape[1]))))

        pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        pred_vertices_smpl = self.smpl(pred_rotmat, pred_betas)

        return pred_vertices, pred_vertices_smpl, camera, pred_rotmat, pred_betas
