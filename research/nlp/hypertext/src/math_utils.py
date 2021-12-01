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
"""math utils"""
import mindspore.common.dtype as mstype
from mindspore.nn import Cell
from mindspore.ops import Log, Sub, clip_by_value

eps = 1e-15


class Artanh(Cell):
    """artanh"""

    def __init__(self):
        """init"""
        super(Artanh, self).__init__()
        self.log = Log()
        self.sub = Sub()

    def construct(self, x):
        """construct fun"""
        x = clip_by_value(x, -1 + eps, 1 - eps)
        out = self.log(1 + x.astype(mstype.float32))
        out = 0.5 * self.sub(out, self.log(1 - x.astype(mstype.float32)))
        return out
