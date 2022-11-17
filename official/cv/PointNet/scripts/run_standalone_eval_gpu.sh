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

if [ $# -ne 3 ]
then 
    echo "Usage: bash scripts/run_standalone_eval_gpu.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]"
exit 1
fi
DATA_PATH=$1
MODEL_PATH=$2
DEVICE_ID=$3
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1
PATH2=$(get_real_path $2)
echo $PATH2
if [ $# == 3 ]; then
    DEVICE_ID=$3
fi


export RANK_SIZE=1

echo "======start training======"

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
nohup python ./eval.py \
  --data_path=$DATA_PATH \
  --device_target="GPU" \
  --model_path=$MODEL_PATH >log_standalone_eval_gpu 2>&1 &
