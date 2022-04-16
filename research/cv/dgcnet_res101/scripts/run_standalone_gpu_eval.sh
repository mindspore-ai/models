#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh [DATA_DIR] [DATA_LIST] [RESTORE_FROM]"
echo "For example: bash run.sh /path/dataset /path/datalist /path/ckpt "
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

export DATA_DIR=$1
export DATA_LIST=$2
export RESTORE_FROM=$3

echo "start evaling"
  python eval.py --data_set cityscapes \
  --data_dir $1 \
  --data_list $2 \
  --restore_from $3 \
  --output_dir "./eval_dual_seg_r101_832"