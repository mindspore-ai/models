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

import mindspore as ms
from mindspore import ops, nn
from mindspore import numpy as msnp


def clip_coordinates(ind, clip_limit):
    return ops.clip_by_value(ind, 0, clip_limit - 1)

class grid_sample(nn.Cell):
    def __init__(self):
        super(grid_sample, self).__init__()
        self.gather = ops.GatherNd()
        self.concat = ops.Concat(1)

    def construct(self, input_tens, grid):
        B, C, H, W = input_tens.shape
        _, IH, IW, _ = grid.shape
        B_ind = ops.cast(msnp.arange(B).repeat(C * IH * IW), ms.int32).reshape((-1, 1))
        C_ind = ops.cast(msnp.arange(C).repeat(IH * IW), ms.int32).reshape((-1, 1))
        C_ind = ops.Tile()(C_ind, (B, 1))

        iy = ops.Tile()((((grid[..., 1] + 1) / 2) * (H - 1)).reshape((-1, 1)), (C, 1))
        ix = ops.Tile()((((grid[..., 0] + 1) / 2) * (W - 1)).reshape((-1, 1)), (C, 1))

        ix_nw = clip_coordinates(ops.floor(ix), W)
        iy_nw = clip_coordinates(ops.floor(iy), H)
        ix_se = clip_coordinates(ix_nw + 1, W)
        iy_se = clip_coordinates(iy_nw + 1, H)

        nw_ind = self.concat((B_ind, C_ind, ops.cast(iy_nw, ms.int32), ops.cast(ix_nw, ms.int32)))
        nw = self.gather(input_tens, nw_ind)
        ne_ind = self.concat((B_ind, C_ind, ops.cast(iy_nw, ms.int32), ops.cast(ix_se, ms.int32)))
        ne = self.gather(input_tens, ne_ind)
        sw_ind = self.concat((B_ind, C_ind, ops.cast(iy_se, ms.int32), ops.cast(ix_nw, ms.int32)))
        sw = self.gather(input_tens, sw_ind)
        se_ind = self.concat((B_ind, C_ind, ops.cast(iy_se, ms.int32), ops.cast(ix_se, ms.int32)))
        se = self.gather(input_tens, se_ind)

        nw_w = ops.absolute(((ix_se - ix) * (iy_se - iy)).reshape((-1,)))
        ne_w = ops.absolute(((ix - ix_nw) * (iy_se - iy)).reshape((-1,)))
        sw_w = ops.absolute(((ix_se - ix) * (iy - iy_nw)).reshape((-1,)))
        se_w = ops.absolute(((ix - ix_nw) * (iy - iy_nw)).reshape((-1,)))

        output = nw_w * nw + ne_w * ne + sw_w * sw + se_w * se

        return output.reshape((B, C, IH, IW))
