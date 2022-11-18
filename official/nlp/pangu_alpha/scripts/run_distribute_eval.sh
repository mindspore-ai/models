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

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
export STRATEGY=$3
export TOKENIZER=$4
export CKPT_PATH=$5
export MODE=$6
export PARAM_INIT_TYPE=$7
export TASK_NAME=$8
export EVAL_DATA_URL=$9


if [ $# != 9 ] ; then
  echo "The input argument number is not sufficient, please see the follow examples:"
  echo "USAGE: bash $0 RANK_SIZE RANK_TABLE_FILE STRATEGY TOKENIZER CKPT_PATH MODE PARAM_INIT_TYPE TASK_NAME EVAL_DATA_URL"
  echo " e.g.: bash $0 8 /root/hccl8p.json /home/ckpts/strategy/strategy.ckpt /home/ckpts/tokenizer/ /home/ckpts/checkpoints 2.6B fp32 c3 /home/data/c3/data/"
  exit 1;
fi



for((i=0;i<$RANK_SIZE;i++));
do
  rm -rf ${execute_path}/device$i/
  mkdir ${execute_path}/device$i/
  cd ${execute_path}/device$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --tokenizer_path=$TOKENIZER --load_ckpt_path=$CKPT_PATH \
                  --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE \
                  --eval_task=$TASK_NAME \
                  --eval_data_url=$EVAL_DATA_URL >log$i.log 2>&1 &
done