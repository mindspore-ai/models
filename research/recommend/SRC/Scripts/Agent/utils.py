# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import ms_function, ops
from numpy import argsort, int32
from numpy.random import rand, randint


@ms_function
def pl_loss(pro, reward):
    return -ops.mean(reward * ops.log(pro + 1e-9))


def generate_path(batchSize, skillNum, path_type, n):
    if path_type in (0, 1):
        origin_path = argsort(rand(batchSize, n))  # 1-N concepts
        if path_type == 1:  # All concepts are grouped by size N
            origin_path += n * randint(0, skillNum // n, (batchSize, 1))
    else:  # 2 or 3
        origin_path = argsort(rand(batchSize, skillNum))  # All concepts should be sorted topN
        if path_type == 2:
            origin_path = origin_path[:, :n]  # Select N concepts randomly and sort them
    return origin_path.astype(int32)
