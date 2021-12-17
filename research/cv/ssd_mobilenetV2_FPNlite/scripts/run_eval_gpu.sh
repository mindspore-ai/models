#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 4 ]
then 
    echo "Usage: bash run_eval_gpu.sh [CONFIG_FILE] [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo realpath -m "$PWD"/"$1"
  fi
}

CONFIG_PATH=$(get_real_path "$1")
DATASET=$2
CKPT_PATH=$(get_real_path "$3")
DEVICE_ID_=$4


if [[ ! -f $CONFIG_PATH ]]
then
    echo "error: CONFIG_FILE=$CONFIG_PATH is not a file"
exit 1
fi

if [ ! -f "$CKPT_PATH" ]
then 
    echo "error: CHECKPOINT_PATH=$CKPT_PATH is not a file"
exit 1
fi 

#ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$DEVICE_ID_
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -- *.sh ./eval
cp -r ../src ./eval
cp -r ../config/*.yaml ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"
python eval.py \
--config_path="$CONFIG_PATH" \
--dataset="$DATASET" \
--checkpoint_path="$CKPT_PATH" \
--run_platform="GPU" \
--device_id="$DEVICE_ID" &> log &
cd ..
