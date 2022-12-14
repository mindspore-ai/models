#!/bin/bash
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
if [ $# -ne 3 ]; then
  echo "Usage: bash scripts/preprocess_data.sh [DATASET_DIR][TRAIN_DIR][TEST_DIR]"
  exit 1
else
  DATASET_DIR=$1
  TRAIN_DIR=$2
  TEST_DIR=$3
  python ./preprocess_data.py --dataset_dir=$DATASET_DIR --train_dir=$TRAIN_DIR --test_dir=$TEST_DIR
fi
