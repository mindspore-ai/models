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

"""init network weights"""

import mindspore.nn as nn
class InitWeights_He():
    """init weights using kaiming normal"""
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform():
    """init weights using XavierUniform"""
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
