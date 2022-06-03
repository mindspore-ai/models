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
"""Bi-directional Gated Feature Fusion"""

from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops


class BiGFF(nn.Cell):
    """Bi-directional Gated Feature Fusion"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.structure_gate = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.Sigmoid()
        )
        self.texture_gate = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.Sigmoid()
        )
        self.structure_gamma = Parameter(ops.Zeros()(1, mstype.float32))
        self.texture_gamma = Parameter(ops.Zeros()(1, mstype.float32))

    def construct(self, texture_feature, structure_feature):
        """construct"""
        energy = ops.Concat(axis=1)((texture_feature, structure_feature))

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
        structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)

        return ops.Concat(axis=1)((texture_feature, structure_feature))
