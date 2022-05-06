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
"""Model layers."""
from mindspore import nn


class Invertible1x1Conv(nn.Cell):
    """
    The layer outputs both the convolution,
    and the log determinant of its weight matrix.
    """
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c,
            out_channels=c,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False,
        )

    def construct(self, z):
        z = self.conv(z)

        return z
