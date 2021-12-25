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
'''
STPM
'''
import mindspore.nn as nn
from src.resnet import resnet18

class STPM(nn.Cell):
    '''STPM
    '''
    def __init__(self):
        super(STPM, self).__init__()
        self.model_t = resnet18(1001)
        self.model_s = resnet18(1001)

    def construct(self, x):
        features_s = self.model_s(x)
        features_t = self.model_t(x)
        return features_s, features_t
