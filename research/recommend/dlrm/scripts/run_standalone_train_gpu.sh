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
echo "Please run the script as:"
echo "bash scripts/run_standalone_train_gpu.sh DEVICE_ID DEVICE_TARGET DATASET_PATH"
echo "for example: bash scripts/run_standalone_train_gpu.sh 0 GPU /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in output.log"


if [ $# -lt 3 ]; then
  echo "Usage: bash scripts/run_standalone_train_gpu.sh DEVICE_ID DEVICE_TARGET DATASET_PATH"
  exit 1
fi

DEVICE_TARGET=$2

if [ "$DEVICE_TARGET" = "GPU" ];
then
  export DEVICE_ID=$1
fi

DATA_URL=$3

if [ ! -d "${DATA_URL}" ]; then
  echo "File or path not exists"
  exit 1
fi

rm auc.log loss.log output.log
rm -rf checkpoint

echo "Setting ulimit number of files to be opened by the process"
ulimit -n 65535

python -u train.py \
    --config_path=gpu_config.yaml \
    --dataset_path=$DATA_URL \
    --ckpt_path="checkpoint" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=$DEVICE_TARGET \
    --do_eval=True > output.log 2>&1 &
