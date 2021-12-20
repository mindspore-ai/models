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
""" Attention Factory
Hacked together by / Copyright 2021 Ross Wightman
"""

from mindspore import nn
from mindspore import ops

from src.models.helpers import make_divisible


class SEModule(nn.Cell):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, act_layer=nn.ReLU, norm_layer=None):
        super(SEModule, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, has_bias=True)
        self.bn = None
        if norm_layer is not None:
            self.bn = norm_layer(rd_channels)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, has_bias=True)
        self.gate = nn.Sigmoid()

    def construct(self, x):
        x_se = ops.ReduceMean(True)(x, (2, 3))

        x_se = self.fc1(x_se)
        if self.bn is not None:
            x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
