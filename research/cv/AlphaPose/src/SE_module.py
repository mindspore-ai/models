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
class SELayer(nn.Cell):
    '''
    SELayer
    '''
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc = nn.SequentialCell(
            [nn.Dense(channel, channel // reduction),
             nn.ReLU(),
             nn.Dense(channel // reduction, channel),
             nn.Sigmoid()]
        )

    def construct(self, x):
        '''
        construct
        '''
        b, c, _, _ = x.shape
        y = self.avg_pool(x, (2, 3)).view((b, c))
        y = self.fc(y).view((b, c, 1, 1))
        return x * y
