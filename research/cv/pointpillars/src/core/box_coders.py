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
"""box coders"""
from mindspore import numpy as mnp
from mindspore import ops
import numpy as np
from src.core import box_np_ops
from src.core import box_ops


class GroundBox3dCoder:
    """ground box 3d coder"""
    def __init__(self, linear_dim=False, vec_encode=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode

    def encode(self, inp, anchors):
        """encode"""
        return box_np_ops.second_box_encode(inp, anchors, self.vec_encode, self.linear_dim)

    def decode(self, inp, anchors):
        """decode"""
        if isinstance(anchors, np.ndarray):
            return box_np_ops.second_box_decode(inp, anchors, self.vec_encode, self.linear_dim)
        return box_ops.second_box_decode(inp, anchors, self.vec_encode, self.linear_dim)

    @property
    def code_size(self):
        """code size"""
        return 8 if self.vec_encode else 7


class BevBoxCoder:
    """bev box coder"""
    def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0, h_fixed=2.0):
        super().__init__()
        self.linear_dim = linear_dim
        self.z_fixed = z_fixed
        self.h_fixed = h_fixed
        self.vec_encode = vec_encode

    def encode(self, inp, anchors):
        """encode"""
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        inp = inp[..., [0, 1, 3, 4, 6]]
        return box_np_ops.bev_box_encode(inp, anchors, self.vec_encode, self.linear_dim)

    def decode(self, inp, anchors):
        """decode"""
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = box_ops.bev_box_decode(inp, anchors, self.vec_encode, self.linear_dim)
        z_fixed = mnp.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
        h_fixed = mnp.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
        return ops.Concat(axis=-1)([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]])

    @property
    def code_size(self):
        """code size"""
        return 6 if self.vec_encode else 5
