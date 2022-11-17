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

# bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]

if [ $# -ne 1 ]; then
    echo "Usage: bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

PATH1=$(get_real_path $1)

if [ ! -f ${PATH1} ]; then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file."
exit 1
fi

cd ..
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

echo 'start training'
for((i=0;i<=$RANK_SIZE-1;i++));
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ./*.py ./train_parallel$i
    cp ./*.yaml ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --is_distributed 1 > ./train$i.log  2>&1 &
    cd ../
done