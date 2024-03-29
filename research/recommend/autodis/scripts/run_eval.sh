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
echo "Please run the script as: "
echo "bash scripts/run_eval.sh [DEVICE_ID] [DEVICE_TARGET] [TEST_DATA_DIR] [CHECKPOINT_PATH]"
echo "for example: bash scripts/run_eval.sh 0 GPU /dataset_path /checkpoint_path"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/eval_output.log"

DEVICE_TARGET=$2
DATA_URL=$(readlink -f "$3")
CHECKPOINT_PATH=$(readlink -f "$4")

DEVICE_TARGET=$2
if [ "$DEVICE_TARGET" = "GPU" ]; then
  export CUDA_VISIBLE_DEVICES=$1
elif [ "$DEVICE_TARGET" = "Ascend" ]; then
  export DEVICE_ID=$1
else
  echo "Unsupported platform:$DEVICE_TARGET"
  exit 1
fi

abs_path=$(readlink -f "$0")
cur_path=$(dirname $abs_path)
cd $cur_path

rm -rf ./eval_$DEVICE_TARGET
mkdir ./eval_$DEVICE_TARGET
cp ../eval.py ./eval_$DEVICE_TARGET
cp ../*.yaml ./eval_$DEVICE_TARGET
cp -r ../src ./eval_$DEVICE_TARGET
cd ./eval_$DEVICE_TARGET || exit

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python -u eval.py \
    --test_data_dir=$DATA_URL \
    --checkpoint_path=$CHECKPOINT_PATH \
    --device_target=$DEVICE_TARGET > ms_log/eval_output.log 2>&1 &
