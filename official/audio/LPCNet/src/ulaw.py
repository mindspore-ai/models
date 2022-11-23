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

import math

import numpy as np

scale = 255.0/32768.0 # 8-bit to 16-bit range ratio
scale_1 = 32768.0/255.0 # 16-bit to 8-bit range ratio


def ulaw2lin(u):
    """ Transform signal from 8-bit mu-law quantized to linear """
    u = u - 128
    s = np.sign(u)
    u = np.abs(u)
    return s*scale_1*(np.exp(u/128.*math.log(256))-1)


def lin2ulaw(x):
    """ Transform signal from linear to 8-bit mu-law quantized """
    s = np.sign(x)
    x = np.abs(x)
    u = (s*(128*np.log(1+scale*x)/math.log(256)))
    u = np.clip(128 + np.round(u), 0, 255)
    return u.astype('int16')
