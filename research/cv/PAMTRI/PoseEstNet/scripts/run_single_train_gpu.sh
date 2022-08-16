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
echo "bash run_single_train_gpu.sh DATA_PATH PRETRAINED_PATH DEVICE_ID"
echo "For example: bash run_single_train_gpu.sh /path/dataset /path/pretrained_path 0 "
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
PRE_CKPT=$(get_real_path $2)
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export DEVICE_ID=$3
export RANK_SIZE=1

echo "Start training on GPU"
python ${PROJECT_DIR}/../train.py --cfg ${PROJECT_DIR}/../config_gpu.yaml \
  --data_dir ${DATA_PATH} \
  --distribute False \
  --device_target GPU \
  --pre_ckpt_path ${PRE_CKPT} > ${PROJECT_DIR}/../train_gpu.log 2>&1 &


