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

import json
import os

import numpy as np


def split(path, out, test_ratio=0.2):
    """
    split dataset
    Args:
        path: path to json format dataset which contains full sample
        out: output dir
        test_ratio: eval ratio
    """
    with open(path, 'r') as f:
        dataset = json.load(f)
        dataset = np.array(dataset)
    np.random.seed(1256)
    dataset_len = len(dataset)
    shuffled_indices = np.random.permutation(dataset_len)
    test_size = int(dataset_len * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    with open(os.path.join(out, "train-dataset.json"), 'w') as f:
        f.write(json.dumps(dataset[train_indices].tolist()))
    with open(os.path.join(out, "eval-dataset.json"), 'w') as f:
        f.write(json.dumps(dataset[test_indices].tolist()))
