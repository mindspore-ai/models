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

self_path=$(cd "$(dirname "$0")" || exit; pwd)
export RANK_SIZE=$1
export HOSTFILE=$2
export STRATEGY=$3
export TOKENIZER=$4
export CKPT_PATH=$5
export MODE=$6
export PARAM_INIT_TYPE=$7
export TASK_NAME=$8
export EVAL_DATA_URL=$9
DEVICE_TARGET=${10}


if [ $# != 10 ] ; then
  echo "The input argument number is not sufficient, please see the follow examples:"
  echo "USAGE: bash $0 RANK_SIZE $HOSTFILE STRATEGY TOKENIZER CKPT_PATH MODE PARAM_INIT_TYPE TASK_NAME EVAL_DATA_URL DEVICE_TARGET"
  echo " e.g.: bash $0 8 /root/hostfile8p /home/ckpts/strategy/strategy.ckpt /home/ckpts/tokenizer/ /home/ckpts/checkpoints 2.6B fp32 c3 /home/data/c3/data/ Ascend"
  exit 1;
fi


mpirun --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x GLOG_v -n $RANK_SIZE --hostfile $HOSTFILE --output-filename log_output --merge-stderr-to-stdout \
  python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --tokenizer_path=$TOKENIZER --load_ckpt_path=$CKPT_PATH \
                  --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE \
                  --eval_task=$TASK_NAME --device_target=$DEVICE_TARGET \
                  --eval_data_url=$EVAL_DATA_URL >log0.log 2>&1 &
