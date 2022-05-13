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
    echo "Usage: bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

DATASET_PATH=$(get_real_path $2)
CHECKPOINT_PATH=$(get_real_path $3)

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: CHECKPOINT_PATH=$CHECKPOINT_PATH is not a file"
    exit 1
fi

export DEVICE_ID=$1
export DEVICE_NUM=1
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

python eval.py --data_url $DATASET_PATH \
  --run_online False --platform GPU \
  --checkpoint $CHECKPOINT_PATH > log 2>&1 &
cd ..
