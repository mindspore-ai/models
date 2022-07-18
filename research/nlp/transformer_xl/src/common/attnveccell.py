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


class AttnVecCell(Cell):

    def __init__(self):
        super(AttnVecCell, self).__init__()
        self.transpose = Transpose()

    def construct(self, attn_prob, w_head_v):
        attn_prob_bnij = self.transpose(attn_prob, (2, 3, 0, 1))
        w_head_v_bnjd = self.transpose(w_head_v, (1, 2, 0, 3))
        attn_vec_bnid = matmul(attn_prob_bnij, w_head_v_bnjd)
        attn_vec_ibnd = self.transpose(attn_vec_bnid, (2, 0, 1, 3))
        return attn_vec_ibnd
