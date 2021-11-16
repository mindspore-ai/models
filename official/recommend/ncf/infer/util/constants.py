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
NCF specific values.
"""
import numpy as np

# Keys for data shards
TRAIN_USER_KEY = "train_user_id"
TRAIN_ITEM_KEY = "train_item_id"
EVAL_USER_KEY = "eval_user_id"
EVAL_ITEM_KEY = "eval_item_id"

USER_MAP = "user_map"
ITEM_MAP = "item_map"

USER_DTYPE = np.int32
ITEM_DTYPE = np.int32

# In both datasets, each user has at least 20 ratings.
MIN_NUM_RATINGS = 20

# The number of negative examples attached with a positive example
# when performing evaluation.
NUM_EVAL_NEGATIVES = 99

# keys for evaluation metrics
TOP_K = 10  # Top-k list for evaluation

BATCH_SIZE = 160000
