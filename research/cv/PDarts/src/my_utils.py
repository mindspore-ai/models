# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define some utils."""
import numpy as np


def print_trainable_params_count(network):
    params = network.trainable_params()
    trainable_params_count = 0
    for param in enumerate(params):
        shape = param[1].data.shape
        size = np.prod(shape)
        trainable_params_count += size
    print("trainable_params_count:" + str(trainable_params_count))


def drop_path(div, mul, x, drop_prob, mask):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        x = div(x, keep_prob)
        x = mul(x, mask)
    return x
