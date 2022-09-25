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

if [ $# != 3 ]
then
    echo "Usage: bash scripts/run_standalone_eval_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH] [DEVICE_ID]"
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_eval_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH] [DEVICE_ID]"
    echo "for example: bash scripts/run_standalone_eval_gpu.sh /path/to/config.yml /path/to/ckpt 0"
    echo "=============================================================================================================="
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)


if [ ! -f $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi


export DEVICE_NUM=1
export DEVICE_ID=$3
export RANK_SIZE=1
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi

mkdir ./eval
cp ./eval.py ./eval
cp -r ./src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"
nohup python eval.py --config=$PATH1 --ckpt_path=$PATH2 --device_id=$DEVICE_ID --device_target='GPU' &> eval.log &

cd ..

