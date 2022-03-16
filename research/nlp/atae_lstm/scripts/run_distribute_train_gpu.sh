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

if [ $# != 2 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_distribute_train_gpu.sh DEVICE_NUM DATA_DIR"
  echo "for example:"
  echo "  bash run_distribute_train_gpu.sh 2 \\"
  echo "      /home/workspace/atae_lstm/data/"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
echo ${BASE_PATH}

ulimit -u unlimited
export RANK_SIZE="$1"
TRAIN_DIR=$(get_real_path "$2")

if [ -d "$BASE_PATH/../train_parallel" ];
then
    rm -rf $BASE_PATH/../train_parallel
fi
mkdir $BASE_PATH/../train_parallel
cd $BASE_PATH/../train_parallel || exit

echo "start distributed training with $RANK_SIZE GPUs."

mpirun --allow-run-as-root -n $RANK_SIZE --merge-stderr-to-stdout \
  python -u $BASE_PATH/../train.py \
    --data_url=$TRAIN_DIR \
    --parallel=True > net_log.log 2>&1 &
