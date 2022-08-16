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
echo "bash run_distribute_train_gpu.sh DATA_PATH pretrain_path RANK_SIZE"
echo "For example: bash run_distribute_train_gpu.sh /path/dataset /path/pretrain_path 8"
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
PRETRAINED_PATH=$(get_real_path $2)
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export DATA_PATH=${DATA_PATH}
export RANK_SIZE=$3

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "Start distribute training on GPU"

mpirun -n $3 --allow-run-as-root python ${PROJECT_DIR}/../train.py --cfg ${PROJECT_DIR}/../config_gpu.yaml \
  --data_dir ${DATA_PATH} \
  --distribute True \
  --device_target GPU \
  --pre_ckpt_path ${PRETRAINED_PATH} > ${PROJECT_DIR}/../distributed_train.log 2>&1 &

