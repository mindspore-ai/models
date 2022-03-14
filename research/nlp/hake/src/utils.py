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
"""Weight init utilities"""

import numpy as np
import mindspore
from mindspore import Tensor, set_seed
from src.config import config

np.random.seed(config.random_seed)
set_seed(config.random_seed)


def uniform_weight(embedding_range, shape):
    uniform = np.random.uniform(-embedding_range, embedding_range, shape)
    return Tensor(uniform, dtype=mindspore.dtype.float32)


def rel_init(embedding_range, shape, hidden_dim):
    uniform = np.random.uniform(-embedding_range, embedding_range, (shape[0], hidden_dim))
    ones = np.ones((shape[0], hidden_dim))
    zeros = np.zeros((shape[0], hidden_dim)) + 0.000001
    rel = np.concatenate((uniform, ones, zeros), axis=1)
    return Tensor(rel, dtype=mindspore.dtype.float32)
