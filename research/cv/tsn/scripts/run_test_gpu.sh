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

if [ $# != 4 ]; then
  echo "Usage: bash run_test_gpu.sh [ROOT_PATH] [CKPT_PATH] [MODALITY] [DEVICE_ID]"
  exit 1
fi

ROOT_PATH=$1
CKPT_PATH=$2
MODALITY=$3
export CUDA_VISIBLE_DEVICES=$4
REPO_PATH=${ROOT_PATH}/tsn
DATA_DIR=${ROOT_PATH}/data/data_extracted/ucf101/tvl1
TEST_LIST=${ROOT_PATH}/data/data_extracted/ucf101/ucf101_val_split_1_rawframes.txt
SCORE_SAVE_PATH=${REPO_PATH}/checkpoint/${MODALITY}/scores_${MODALITY}

python ${REPO_PATH}/test_net.py \
  --platform=GPU \
  --modality=${MODALITY} \
  --weights=${CKPT_PATH} \
  --dataset_path=${DATA_DIR} \
  --test_list=${TEST_LIST} \
  --save_scores=${SCORE_SAVE_PATH} > ${MODALITY}_test.log 2>&1 &
echo "Testing background..."