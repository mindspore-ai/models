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
Functions for slimmable operations
"""
import mindspore.nn as nn

class SwitchableBatchNorm2d(nn.Cell):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features_list[1])

    def construct(self, input0):
        return self.bn(input0)

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list,
                 out_channels_list,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(SlimmableConv2d, self).__init__(in_channels_list[1],
                                              out_channels_list[1],
                                              kernel_size,
                                              stride=stride,
                                              pad_mode='pad',
                                              padding=padding,
                                              dilation=dilation,
                                              group=1,
                                              has_bias=bias)

class SlimmableLinear(nn.Dense):
    def __init__(self,
                 in_features_list,
                 out_features_list,
                 bias=True):
        super(SlimmableLinear, self).__init__(in_features_list[1],
                                              out_features_list[1],
                                              has_bias=bias)

def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]
