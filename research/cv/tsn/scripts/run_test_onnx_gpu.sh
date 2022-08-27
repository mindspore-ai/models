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
echo "bash run_test_onnx_gpu.sh [ONNX_PATH] [MODALITY] [DATA_DIR] [TEST_LIST] [SCORE_SAVE_PATH]"
echo "For example:"
echo "bash run_test_onnx_gpu.sh /hy-tmp/cyj/tsn/tsn_RGB.onnx RGB /hy-tmp/ucf101/ /hy-tmp/cyj/ucf101_val_split_1_rawframes.txt /hy-tmp/cyj/tsn/scores_RGB_onnx"
echo "It is better to use the ABSOLUTE path."
echo "=============================================================================================================="

if [ $# != 5 ]; then
  echo "Usage: bash run_test_onnx_gpu.sh [ONNX_PATH] [MODALITY] [DATA_DIR] [TEST_LIST] [SCORE_SAVE_PATH]"
  exit 1
fi


ONNX_PATH=$1
MODALITY=$2
DATA_DIR=$3
TEST_LIST=$4
SCORE_SAVE_PATH=$5

python ../test_net_onnx.py \
  --platform=GPU \
  --modality=${MODALITY} \
  --onnx_path=${ONNX_PATH} \
  --dataset_path=${DATA_DIR} \
  --test_list=${TEST_LIST} \
  --save_scores=${SCORE_SAVE_PATH} > ${MODALITY}_test.log 2>&1 &
echo "Testing background..."