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
"""Common class for model."""
import mindspore.nn as nn

def repeat_3d_tuple(origin):
    """
    expand tuple (x,y,z) to (x,x,y,y,z,z)
    """
    assert isinstance(origin, (tuple, list)) and len(origin) == 3, 'input tuple for repeat is not a 3d tuple'
    return tuple([origin[0], origin[0], origin[1], origin[1], origin[2], origin[2]])

class Ops2Cell(nn.Cell):
    """
    capsulate ops to a Cell
    Args:
        action: ops.xx, e.g. ops.MaxPool3D
    Inputs:
        x(Tensor): input
    """
    def __init__(self, action):
        super().__init__()
        self.action = action

    def construct(self, x):
        return self.action(x)
