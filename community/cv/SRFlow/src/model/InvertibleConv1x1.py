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
"""
The Invconvlution Layer of Glow
"""

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops


class InvertibleConv1x1(nn.Cell):
    """
    The Invconvlution Layer of Glow
    """
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = Tensor(np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32))
        self.weight = Parameter(default_input=w_init, name='weight')
        self.num_channels = num_channels

    def construct(self, x, logdet=None):
        conv2d = ops.Conv2D(out_channel=self.num_channels, kernel_size=(1, 1), stride=1)
        reshape = ops.Reshape()
        weight = reshape(self.weight, (self.num_channels, self.num_channels, 1, 1))
        z = conv2d(x, weight)
        # logdet += torch.slogdet(self.weight)[1] * x[2] * x[3]
        return z, logdet
