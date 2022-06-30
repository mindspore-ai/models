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

import mindspore as ms
from mindspore import ops, Tensor, nn, Parameter
from mindspore.ops import constexpr
import numpy as np
from scipy.sparse import coo_matrix

try:
    import cPickle as pickle
except ImportError:
    import pickle

from models.geometric_layers import rodrigues
import utils.config as cfg

@constexpr
def generate_Tensor_att():
    return Tensor([0, 0, 0, 1], dtype=ms.float32)

def generate_Tensor_np(temp):
    return Tensor(temp)

def generate_Tensor(temp):
    return Tensor(temp, dtype=ms.float32)

def generate_Tensor_int(temp):
    return Tensor(temp, dtype=ms.int64)

def generate_Tensor_at(temp):
    return Tensor.from_numpy(temp.astype(np.int64))

def generate_Tensor_atf(temp):
    return Tensor.from_numpy(temp.astype('float32'))


def sparse_to_dense(x, y, shape):
    return coo_matrix((y, x), shape, dtype=np.float).todense()

def to_int(temp):
    return int(temp)

def Einsum(x, y):
    return Tensor(np.einsum('bik,ji->bjk', x, y), dtype=ms.float32)

class SMPL(nn.Cell):
    def __init__(self, model_file=cfg.SMPL_FILE):
        super(SMPL, self).__init__()
        with open(model_file, "rb") as f:
            smpl_model = pickle.load(f, encoding="iso-8859-1")
        J_regressor = smpl_model["J_regressor"].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = generate_Tensor([row, col])
        v = generate_Tensor(data)
        J_regressor_shape = [24, 6890]

        a1 = i.asnumpy()
        v1 = v.asnumpy()
        J_regressor = sparse_to_dense(a1, v1, J_regressor_shape)
        J_regressor = generate_Tensor(J_regressor)

        self.J_regressor = Parameter(J_regressor, name="J_regressor", requires_grad=False)

        weights = generate_Tensor(smpl_model['weights'])
        self.weights = Parameter(weights, name="weights", requires_grad=False)

        posedirs = generate_Tensor(smpl_model['posedirs'])
        self.posedirs = Parameter(posedirs, name="posedirs", requires_grad=False)

        v_template = generate_Tensor(smpl_model['v_template'])
        self.v_template = Parameter(v_template, name="v_template", requires_grad=False)

        shapdirs = generate_Tensor(np.array(smpl_model['shapedirs']))
        self.shapedirs = Parameter(shapdirs, name="shapedirs", requires_grad=False)

        faces = generate_Tensor_at(smpl_model['f'])
        self.faces = Parameter(faces, name="faces", requires_grad=False)

        kintree_table = generate_Tensor_at(smpl_model['kintree_table'])
        self.kintree_table = Parameter(kintree_table, name="kintree_table", requires_grad=False)

        id_to_col = {to_int(self.kintree_table[1, i].asnumpy()): i for i in range(self.kintree_table.shape[1])}

        parent = generate_Tensor_int([id_to_col[to_int(kintree_table[0, it].asnumpy())]
                                      for it in range(1, kintree_table.shape[1])])
        self.parent = Parameter(parent, name="parent", requires_grad=False)

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = ms.numpy.zeros(self.pose_shape)
        self.beta = ms.numpy.zeros(self.beta_shape)
        self.translation = ms.numpy.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        J_regressor_extra = generate_Tensor_atf(np.load(cfg.JOINT_REGRESSOR_TRAIN_EXTRA))
        self.J_regressor_extra = Parameter(J_regressor_extra, name="J_regressor_extra", requires_grad=False)

        self.joints_idx = cfg.JOINTS_IDX

        h36m_regressor_cmr = generate_Tensor(np.load(cfg.JOINT_REGRESSOR_H36M))
        self.h36m_regressor_cmr = Parameter(h36m_regressor_cmr, name="h36m_regressor_cmr", requires_grad=False)

        lsp_regressor_cmr = generate_Tensor(np.load(cfg.JOINT_REGRESSOR_H36M))
        self.lsp_regressor_cmr = Parameter(lsp_regressor_cmr[cfg.H36M_TO_J14], name="lsp_regressor_cmr",
                                           requires_grad=False)

        lsp_regressor_eval = generate_Tensor(np.load(cfg.LSP_REGRESSOR_EVAL))
        self.lsp_regressor_eval = Parameter(lsp_regressor_eval.transpose(1, 0), name="lsp_regressor_eval",
                                            requires_grad=False)

        # We hope the training and evaluation regressor for the lsp joints to be consistent,
        # so we replace parts of the training regressor.
        train_regressor = ops.Concat(0)((self.J_regressor, self.J_regressor_extra))
        train_regressor = train_regressor[[cfg.JOINTS_IDX]].copy()
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        train_regressor[idx] = self.lsp_regressor_eval
        self.train_regressor = Parameter(train_regressor, name="train_regressor", requires_grad=False)

    def construct(self, pose, beta):
        batch_size = pose.shape[0]
        if batch_size == 0:
            return ops.Zeros()((0, 6890, 3), pose.dtype)
        v_template = self.v_template[None, :]

        broadcast_to = ops.BroadcastTo((batch_size, -1, -1))
        shapedirs = broadcast_to(self.shapedirs.view(-1, 10)[None, :])

        beta = beta[:, :, None]
        v_shaped = ops.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template

        J = []
        for i in range(batch_size):
            J.append(ops.matmul(self.J_regressor, v_shaped[i]))
        J = ops.Stack(axis=0)(J)

        pose_cube = pose.view(-1, 3)
        R = pose if (pose.ndim == 4) else rodrigues(pose_cube).view(batch_size, 24, 3, 3)

        I_cube = ops.Eye()(3, 3, pose.dtype)[None, None, :]

        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        broadcast_to = ops.BroadcastTo((batch_size, -1, -1))
        posedirs = broadcast_to(self.posedirs.view(-1, 207)[None, :])
        v_posed = v_shaped + ops.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)

        J_ = J.copy()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = ops.Concat(axis=-1)([R, J_[:, :, :, None]])

        broadcast_too = ops.BroadcastTo((batch_size, 24, -1, -1))
        pad = generate_Tensor_att()

        pad_row = broadcast_too(pad.view(1, 1, 1, 4))

        G_ = ops.Concat(axis=2)([G_, pad_row])
        G = G_.copy()

        for i in range(1, 24):
            G[:, i, :, :] = ops.matmul(G[:, self.parent[i - 1], :, :], G_[:, i, :, :])

        rest = ops.Concat(axis=2)((J, ops.Zeros()((batch_size, 24, 1), pose.dtype))).view(batch_size, 24, 4, 1)

        zeros = ops.Zeros()((batch_size, 24, 4, 3), pose.dtype)
        rest = ops.Concat(axis=-1)((zeros, rest))
        rest = ops.matmul(G, rest)
        G = G - rest
        T = ops.matmul(self.weights, G.transpose(1, 0, 2, 3).view(24, -1)).\
            view(6890, batch_size, 4, 4).transpose(1, 0, 2, 3)

        rest_shape_h = ops.Concat(axis=-1)([v_posed, ops.OnesLike()(v_posed)[:, :, [0]]])
        v = ops.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]

        return v

    # The function used for outputting the 24 training joints.
    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = Einsum(vertices.asnumpy(), self.J_regressor.asnumpy())
        joints_extra = Einsum(vertices.asnumpy(), self.J_regressor_extra.asnumpy())
        joints = ops.Concat(axis=1)((joints, joints_extra))
        joints = joints[:, cfg.JOINTS_IDX]
        return joints

    # The function used for getting 38 joints.
    def get_full_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """

        joints = Einsum(vertices, self.J_regressor_temp)
        joints_extra = Einsum(vertices, self.J_regressor_extra_temp)
        joints = ops.Concat(axis=1)((joints, joints_extra))
        return joints

    # Get 14 lsp joints use the joint regressor.
    def get_lsp_joints(self, vertices):
        joints = ops.matmul(self.lsp_regressor_cmr[None, :], vertices)
        return joints

    # Get the joints defined by SMPL model.
    def get_smpl_joints(self, vertices):
        """
        This method is used to get the SMPL model joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)ijk,ikn->ijn
        """
        joints = Einsum(vertices, self.J_regressor_temp)
        return joints

    # Get 24 training joints using the evaluation LSP joint regressor.
    def get_train_joints(self, vertices):
        """
        This method is used to get the training 24 joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = ops.matmul(self.train_regressor[None, :], vertices)
        return joints

    # Get 14 lsp joints for the evaluation.
    def get_eval_joints(self, vertices):
        """
        This method is used to get the 14 eval joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 14, 3)
        """
        joints = ops.matmul(self.lsp_regressor_eval[None, :], vertices)
        return joints
