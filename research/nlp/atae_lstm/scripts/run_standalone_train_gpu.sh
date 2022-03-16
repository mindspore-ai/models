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

if [ $# != 1 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_standalone_train_gpu.sh DATA_DIR"
  echo "for example:"
  echo "  bash run_standalone_train_gpu.sh \\"
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
TRAIN_DIR=$(get_real_path "$1")

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

echo "start standalone training on GPU."

python -u $BASE_PATH/../train.py \
    --data_url=$TRAIN_DIR \
    --parallel=False > net_log.log 2>&1 &
