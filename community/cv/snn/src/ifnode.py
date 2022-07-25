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
"""IFNode."""
import mindspore.nn as nn
from mindspore import ops
import mindspore


class relusigmoid(nn.Cell):
    """
    custom surrogate function for integrate and fire cell
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = ops.Sigmoid()
        self.greater = ops.Greater()

    def construct(self, x):
        spike = self.greater(x, 0)
        return spike.astype(mindspore.float32)

    def bprop(self, x, out, dout):
        sgax = self.sigmoid(x * 5.0)
        grad_x = dout * (1 - sgax) * sgax * 5.0
        # must be a tuple
        return (grad_x,)

class IFNode(nn.Cell):
    """
    integrate and fire cell for GRAPH mode, it will output spike value
    """
    def __init__(self, v_threshold=1.0, fire=True, surrogate_function=relusigmoid()):
        super().__init__()
        self.v_threshold = v_threshold
        self.fire = fire
        self.surrogate_function = surrogate_function

    def construct(self, x, v):
        """neuronal_charge: v need to do add"""
        v = v + x
        if self.fire:
            spike = self.surrogate_function(v - self.v_threshold) * self.v_threshold
            v -= spike
            return spike, v
        return v, v
