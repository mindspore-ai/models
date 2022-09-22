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

if [ $# != 2 ] && [ $# != 3 ]; then
  echo "================================================================================================================"
  echo "Please run the script as: "
  echo "bash run_distributed_train_gpu.sh MINDRECORD_DIR NUM_DEVICES LOAD_CHECKPOINT_PATH(optional)"
  echo "for example: bash run_distributed_train_gpu.sh /path/mindrecord_dataset /path/load_ckpt"
  echo "if no ckpt, just run: bash run_distributed_train_gpu.sh /path/mindrecord_dataset"
  echo "It is better to use the absolute path."
  echo "================================================================================================================"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

MINDRECORD_DIR=$(get_real_path "$1")

export RANK_SIZE=$2

if [ $# == 3 ];
then
    LOAD_CHECKPOINT_PATH=$3
else
    LOAD_CHECKPOINT_PATH=""
fi

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

CONFIG=$(get_real_path "$PROJECT_DIR/../centernetdet_gpu_config.yaml")

train_path='./Train_parallel'
if [ -d ${train_path} ]; then
  rm -rf ${train_path}
fi
mkdir -p ${train_path}
cp *.py  ${train_path}
cp *.yaml ${train_path}
cp -r src  ${train_path}
cd ${train_path} || exit

echo "Start distributed training with $RANK_SIZE GPUs."

mpirun --allow-run-as-root -n $RANK_SIZE --merge-stderr-to-stdout \
    python -u ${PROJECT_DIR}/../train.py  \
        --config_path $CONFIG \
        --distribute=true \
        --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
        --mindrecord_dir=$MINDRECORD_DIR > training_log.txt 2>&1 &