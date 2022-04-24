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
"""anchor generator"""
import numpy as np
from src.core import box_np_ops


class AnchorGeneratorStride:
    """anchor generator stride"""
    def __init__(self,
                 sizes=(1.6, 3.9, 1.56),
                 anchor_strides=(0.4, 0.4, 1.0),
                 anchor_offsets=(0.2, -39.8, -1.78),
                 rotations=(0, np.pi / 2),
                 class_id=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._dtype = dtype
        self._class_id = class_id
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_id(self):
        """class id"""
        return self._class_id

    @property
    def match_threshold(self):
        """match threshold"""
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        """unmatch threshold"""
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        """num anchors per localization"""
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        """generate"""
        return box_np_ops.create_anchors_3d_stride(feature_map_size, self._sizes,
                                                   self._anchor_strides, self._anchor_offsets,
                                                   self._rotations, self._dtype)
