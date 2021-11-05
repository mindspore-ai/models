# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Rounding operations.
"""
import mindspore.nn as nn
from mindspore.ops import operations as ops


class RoundQuant(nn.Cell):
    """
    Rounding operation: discretize floating points to certain bins.
    """
    def __init__(self, inverse_bin_width):
        super(RoundQuant, self).__init__()
        self.inverse_bin_width = float(inverse_bin_width)
        self.round = ops.Round()

    def construct(self, x):
        """construct"""
        h = x * self.inverse_bin_width
        h = self.round(h)
        h = h / self.inverse_bin_width
        return h
