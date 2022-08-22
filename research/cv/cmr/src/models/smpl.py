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
from mindspore import nn, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore import ops
from mindspore import numpy as np

from src import config as cfg
from src.models.geometric_layers import rodrigues


class SMPL(nn.Cell):

    def __init__(self):
        """
        Define the SMPL model and load the parameters from the checkpoint later
        """
        super(SMPL, self).__init__()
        self.J_regressor = Parameter(Tensor(np.ones((24, 6890))), name='J_regressor', requires_grad=False)
        self.weights = Parameter(Tensor(np.ones((6890, 24))),
                                 name='weights', requires_grad=False)
        self.posedirs = Parameter(Tensor(np.ones((6890, 3, 207))),
                                  name='posedirs', requires_grad=False)
        self.v_template = Parameter(Tensor(np.ones((6890, 3))),
                                    name='v_template', requires_grad=False)
        self.shapedirs = Parameter(Tensor(np.ones((6890, 3, 10)), dtype=mindspore.float32),
                                   name='shapedirs', requires_grad=False)
        self.faces = Parameter(Tensor(np.ones((13776, 3))),
                               name='faces', requires_grad=False)
        self.parent = cfg.PARENT

        self.J_regressor_extra = Parameter(Tensor(np.ones((14, 6890))),
                                           name='J_regressor_extra', requires_grad=False)
        self.joints_idx = cfg.JOINTS_IDX

        self.pad_row = Tensor([0, 0, 0, 1], dtype=mindspore.float32).view(1, 1, 1, 4)
        self.matmul = P.MatMul()
        self.stack = P.Stack(axis=0)
        self.concat_op__1 = P.Concat(axis=-1)
        self.concat_op_2 = P.Concat(axis=2)

    def construct(self, pose: Tensor, beta: Tensor):
        """
        :param pose: shape = (B, 24, 3, 3) or (B, 72)
        :param beta: shape = (B, 10)
        :return: shape = (B, 6890, 3)
        """
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 10)
        shapedirs = np.broadcast_to(shapedirs, (batch_size, shapedirs.shape[0], shapedirs.shape[1]))
        beta = beta[:, :, None]
        v_shaped = ops.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template

        J = []
        for i in range(batch_size):
            J.append(self.matmul(self.J_regressor, v_shaped[i]))
        J = self.stack(J) # shape = (bs, 24, 3)

        R = None
        # input it rotmat: (bs,24,3,3)
        if pose.ndim == 4:
            R = pose

        # input it rotmat: (bs,72)
        elif pose.ndim == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)

        I_cube = np.eye(3)[None, None, :]
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)[None, :]
        posedirs = posedirs.repeat(batch_size, axis=0)
        v_posed = v_shaped + ops.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.copy()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = self.concat_op__1((R, J_[:, :, :, None]))

        pad_row = self.pad_row.repeat(batch_size, axis=0).repeat(24, axis=1)
        G_ = self.concat_op_2((G_, pad_row))
        G = [G_[:, 0].copy()]
        for i in range(1, 24):
            G.append(ops.matmul(G[self.parent[i-1]], G_[:, i, :, :]))
        G = np.stack(G, axis=1)

        rest = self.concat_op_2((J, np.zeros((batch_size, 24, 1)))).view(batch_size, 24, 4, 1)
        zeros = np.zeros((batch_size, 24, 4, 3))
        rest = self.concat_op__1((zeros, rest))
        rest = ops.matmul(G, rest)
        G = G - rest
        T = self.matmul(self.weights, G.transpose(1, 0, 2, 3).view(24, -1))\
                        .view(6890, batch_size, 4, 4).transpose(1, 0, 2, 3)
        rest_shape_h = self.concat_op__1((v_posed, np.ones_like(v_posed)[:, :, [0]]))
        v = ops.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def batch_matmul(self, matrix_1: Tensor, matrix_2: Tensor):
        """
        :param matrix_1: size = (B, 6890, 3)
        :param matrix_2: size = (38, 6890)
        :return: size = (B, 38, 3)
        """
        batchmatmul = P.BatchMatMul()
        batch_size = matrix_1.shape[0]
        matrix_2 = np.broadcast_to(matrix_2, (batch_size, matrix_2.shape[0], matrix_2.shape[1]))
        output = batchmatmul(matrix_2, matrix_1)

        return output

    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = ops.matmul(self.J_regressor, vertices)
        joints_extra = ops.matmul(self.J_regressor_extra, vertices)
        joints = np.concatenate((joints, joints_extra), axis=1)[:, cfg.JOINTS_IDX]
        return joints
