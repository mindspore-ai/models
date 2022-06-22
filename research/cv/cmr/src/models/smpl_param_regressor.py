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
import mindspore.nn as nn
from mindspore import Tensor, numpy as np
from mindspore import ops
import numpy
from src.models.layers import FCResBlock, FCBlock


class SMPLParamRegressor(nn.Cell):
    def __init__(self):
        super(SMPLParamRegressor, self).__init__()
        # 1723 is the number of vertices in the subsampled SMPL mesh
        self.layers = nn.SequentialCell([
            FCBlock(1723 * 6, 1024),
            FCResBlock(1024, 1024),
            FCResBlock(1024, 1024),
            nn.Dense(1024, 24 * 3 * 3 + 10)
        ])

        self.matmul = ops.MatMul()

    def construct(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723, 6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        rotmat = x[:, :24*3*3].view(-1, 24, 3, 3)
        betas = x[:, 24*3*3:]
        rotmat = rotmat.view(-1, 3, 3)
        U, V = batch_svd(rotmat)
        rotmat = ops.matmul(U, V.transpose((0, 2, 1)))
        det = numpy.zeros((rotmat.shape[0], 1, 1))
        for i in range(rotmat.shape[0]):
            det[i] = numpy.linalg.det(rotmat[i].asnumpy())
        det = Tensor(det, dtype=mindspore.float32)
        det = ops.stop_gradient(det)
        rotmat = rotmat * det
        rotmat = rotmat.view(batch_size, 24, 3, 3)
        return rotmat, betas


def batch_svd(A):
    """
    Compute SVD value on a batch of squared 3*3 matrix.
    :param A: shape = (N, 3, 3)
    """
    U_list = []
    V_list = []
    for i in range(A.shape[0]):
        _, U, V = ops.svd(A[i])
        U_list.append(U)
        V_list.append(V)
    U = np.stack(U_list, axis=0)
    V = np.stack(V_list, axis=0)
    return U, V
