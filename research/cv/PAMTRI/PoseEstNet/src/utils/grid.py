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
"""grid"""
import math
import numpy as np

def make_grid(_input, nrow=8, padding=2):
    """make_grid"""
    low = np.min(_input)
    high = np.max(_input)
    _input = np.clip(_input, a_min=low, a_max=high)
    _input = _input - low
    _max = high - low
    if _max < 1e-5:
        _input = _input / 1e-5
    else:
        _input = _input / _max

    if _input.shape[0] == 1:
        return np.squeeze(_input)

    nmaps = _input.shape[0]
    if nrow < nmaps:
        xmaps = nrow
    else:
        xmaps = nmaps
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(_input.shape[2] + padding), int(_input.shape[3] + padding)
    num_channels = _input.shape[1]

    grip = np.zeros((num_channels, height * ymaps + padding, width * xmaps + padding))

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grip[:, y * height + padding : y * height + height, :] \
                [:, :, x * width + padding : x * width + width] = _input[k]
            k = k + 1
    return grip

def permute(_input):
    """permute"""
    _input = _input * 255
    _input = np.clip(_input, a_min=0, a_max=255)
    _input = _input.astype(np.uint8)
    _input = np.transpose(_input, (1, 2, 0))

    return _input
