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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval.sh DATA_PATH TEST_PATH CKPT_PATH DEVICE_ID"
echo "For example: bash run_eval.sh ./BraST17/HGG ./src/test.txt ./*.ckpt 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

if [ $# != 4 ]
then
  echo "Usage: bash run_eval.sh DATA_PATH TEST_PATH CKPT_PATH DEVICE_ID"
exit 1
fi

DATA_PATH=$1
TEST_PATH=$2
CKPT_PATH=$3
DEVICE_ID=$4

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export DEVICE_ID=$DEVICE_ID
export RANK_ID=$DEVICE_ID
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ..
env > env.log
echo "Eval begin"
python eval.py \
--data_path "$DATA_PATH" \
--test_path "$TEST_PATH" \
--ckpt_path "$CKPT_PATH" \
--correction True \
--device_id "$DEVICE_ID" \
> eval.log 2>&1 &
echo "Evaling. Check it at eval.log"