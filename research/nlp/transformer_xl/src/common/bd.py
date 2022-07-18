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

from mindspore.nn import Cell
from mindspore.ops import Transpose, matmul, ExpandDims, Tile


class BdCell(Cell):

    def __init__(self):
        super(BdCell, self).__init__()
        self.transpose = Transpose()
        self.expandDims = ExpandDims()
        self.tile = Tile()

    def construct(self, rr_head_q, r_head_k):
        rr_head_q_bnid = self.transpose(rr_head_q, (1, 2, 0, 3))
        r_head_k_ndj = self.transpose(r_head_k, (1, 2, 0))
        r_head_k_1ndj = self.expandDims(r_head_k_ndj, 0)
        r_head_k_bndj = self.tile(r_head_k_1ndj, (rr_head_q_bnid.shape[0], 1, 1, 1))
        BD_bnij = matmul(rr_head_q_bnid, r_head_k_bndj)
        BD_ijbn = self.transpose(BD_bnij, (2, 3, 0, 1))
        return BD_ijbn
