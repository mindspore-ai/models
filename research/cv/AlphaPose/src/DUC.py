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
'''
Alphapose network
'''
import mindspore.nn as nn
import mindspore.ops as ops
class DUC(nn.Cell):
    '''
    Initialize: inplanes, planes, upython
    pscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, pad_mode='pad', padding=1, has_bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.shuffle = ops.DepthToSpace(upscale_factor)

    def construct(self, x):
        '''
        construct
        '''
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.shuffle(x)
        return x
