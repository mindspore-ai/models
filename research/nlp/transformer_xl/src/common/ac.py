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
from mindspore.ops import Transpose, matmul


class AcCell(Cell):

    def __init__(self):
        super(AcCell, self).__init__()
        self.transpose = Transpose()

    def construct(self, rw_head_q, w_head_k):
        rw_head_q_bnid = self.transpose(rw_head_q, (1, 2, 0, 3))
        w_head_k_bndj = self.transpose(w_head_k, (1, 2, 3, 0))
        AC_bnij = matmul(rw_head_q_bnid, w_head_k_bndj)
        AC_ijbn = self.transpose(AC_bnij, (2, 3, 0, 1))
        return AC_ijbn
