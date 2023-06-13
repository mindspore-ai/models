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

echo "Please run the script as: "
echo "bash run_standalone_train_ascend.sh [DATASET_PATH] [RANK_TABLE_FILE]"
echo "for example: bash run_standalone_train_ascend.sh ~/NYU ~/rank_table_8pcs.json"
echo "After running the script, the network runs in the background, The log will be generated in train_8p.log"

export DATASET_PATH=$1
export RANK_TABLE_FILE=$2
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
export RANK_SIZE=8

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    mkdir device$i/src
    cp ./../train.py ./device$i
    cp ./../src/*.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --is_distributed 1 --data_url $DATASET_PATH > train_8p.log 2>&1 &
    cd ../
done

rm -rf device0
mkdir device0
mkdir device0/src
cp ./../train.py ./device0
cp ./../src/*.py ./device0/src
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python train.py --is_distributed 1 --data_url $DATASET_PATH > train_8p.log 2>&1 &

