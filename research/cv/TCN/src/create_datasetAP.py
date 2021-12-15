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
"""
######################## create and save adding problem dataset ########################
"""
import os

import numpy as np


def create_dataset(N, seq_length, mode='train'):
    """create and save adding problem dataset"""
    np.random.seed(0)
    X_num = np.random.rand(N, 1, seq_length)
    X_num = X_num.astype(np.float32)
    X_mask = np.zeros((N, 1, seq_length), dtype=np.float32)
    Y = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, 0, positions[0]] = 1
        X_mask[i, 0, positions[1]] = 1
        Y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    X = np.concatenate((X_num, X_mask), axis=1)
    X_path = os.path.join('../data/AddProb/train', 'traindata.bin')
    Y_path = os.path.join('../data/AddProb/train', 'trainlabel.bin')
    if mode == 'test':
        X_path = os.path.join('../data/AddProb/test', 'testdata.bin')
        Y_path = os.path.join('../data/AddProb/test', 'testlabel.bin')
    X.tofile(X_path)
    Y.tofile(Y_path)

    return 0
