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

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export STRATEGY=$1
export TOKENIZER=$2
export CKPT_PATH=$3
export CKPT_NAME=$4
export MODE=$5
export PARAM_INIT_TYPE=fp16
export DEVICE_TARGET=$6 # Ascend or GPU
export TASK_NAME=$7
export EVAL_DATA_URL=$8

if [ $# != 8 ] ; then
  echo "The input argument number is not sufficient, please see the follow examples:"
  echo "USAGE: bash $0 STRATEGY TOKENIZER CKPT_PATH CKPT_NAME MODE DEVICE_TARGET TASK_NAME EVAL_DATA_URL"
  echo " e.g.: bash $0 /home/ckpts/strategy/strategy.ckpt /home/ckpts/tokenizer/ /home/ckpts/checkpoints 2.6B Ascend fp32 c3 /home/data/c3/data/"
  exit 1;
fi


rm -rf ${execute_path}/device0/
mkdir ${execute_path}/device0/
cd ${execute_path}/device0/ || exit
python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --tokenizer_path=$TOKENIZER --load_ckpt_path=$CKPT_PATH \
                --load_ckpt_name=$CKPT_NAME --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE \
                --distribute=false  --device_target=$DEVICE_TARGET \
                --eval_task=$TASK_NAME \
                --eval_data_url=$EVAL_DATA_URL \
                --tokenizer_path=$TOKENIZER >log0.log 2>&1 &
