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
import mindspore
from mindspore import Tensor, nn, Parameter, ops
from mindspore.ops import operations as P

import scipy.sparse
import numpy as np


def adjmat_(adjmat, nsize=1):
    """Create row-normalized dense graph adjacency matrix."""
    adjmat = scipy.sparse.csc_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    return adjmat.todense()


def get_graph_params_dense(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices in dense format."""
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    U = data['U']
    D = data['D']
    A = [Tensor(adjmat_(a, nsize=nsize), dtype=mindspore.float32) for a in data['A']]
    U = [Tensor(u.todense(), dtype=mindspore.float32) for u in U]
    D = [Tensor(d.todense(), dtype=mindspore.float32) for d in D]
    return A, U, D


class MeshCell(nn.Cell):
    def __init__(self, smpl, num_downsampling=1):
        super(MeshCell, self).__init__()
        self.A0 = Parameter(Tensor(np.ones((6890, 6890)).astype(np.float32)), name='A0', requires_grad=False)
        self.A1 = Parameter(Tensor(np.ones((1723, 1723)).astype(np.float32)), name='A1', requires_grad=False)
        self.A2 = Parameter(Tensor(np.ones((431, 431)).astype(np.float32)), name='A2', requires_grad=False)
        self.U0 = Parameter(Tensor(np.ones((6890, 1723)).astype(np.float32)), name='U0', requires_grad=False)
        self.U1 = Parameter(Tensor(np.ones((1723, 431)).astype(np.float32)), name='U1', requires_grad=False)
        self.D0 = Parameter(Tensor(np.ones((1723, 6890)).astype(np.float32)), name='D0', requires_grad=False)
        self.D1 = Parameter(Tensor(np.ones((431, 1723)).astype(np.float32)), name='D1', requires_grad=False)

        self.A = []
        self.U = []
        self.D = []

        self.num_downsampling = num_downsampling

        # load template vertices from SMPL and normalize them
        ref_vertices = smpl.v_template
        center = 0.5 * (ref_vertices.max(axis=0) + ref_vertices.min(axis=0))[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max()

        self._ref_vertices = ref_vertices
        self.faces = Tensor(smpl.faces, dtype=mindspore.int32)

        self.matmul = P.MatMul()
        self.stack = P.Stack(axis=0)

    def construct(self, x):
        """
        Upsample mesh.
        :param x: shape = (B, 1723, 3)
        :return: pred_vertices: shape = (B, 6890, 3)
        """
        res = ops.matmul(self.U[0], x)
        return res

    @property
    def adjmat(self):
        return self.A[self.num_downsampling]

    @property
    def ref_vertices(self):
        ref_vertices = self._ref_vertices
        for i in range(self.num_downsampling):
            ref_vertices = self.matmul(self.D[i], ref_vertices)

        return ref_vertices

    def update_paramter(self):
        self.A = [Tensor(self.A0), Tensor(self.A1), Tensor(self.A2)]
        self.U = [Tensor(self.U0), Tensor(self.U1)]
        self.D = [Tensor(self.D0), Tensor(self.D1)]
