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
import math
import numpy as np


# https://stackoverflow.com/a/41206290/3839572
def normal_round(n):
    if isinstance(n, (list, np.ndarray)):
        if isinstance(n, list):
            temp = np.array(n)
        else:
            temp = n

        for idx, value in np.ndenumerate(temp):
            if value - math.floor(value) < 0.5:
                temp[idx] = math.floor(value)
            temp[idx] = math.ceil(value)
        return temp
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)
