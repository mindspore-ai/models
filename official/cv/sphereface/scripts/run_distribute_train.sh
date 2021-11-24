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


echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh"
echo "For example: bash run_distribute_train.sh"
echo "=============================================================================================================="
set -e
export RANK_TABLE_FILE=/data/hccl_8p_01234567_127.0.0.1.json
export RANK_SIZE=8
EXEC_PATH=$(pwd)

echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ../*.py ./device$i
    cp ../*.yaml ./device$i
    cp -r ../src ./device$i/
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py > output.log 2>&1 &
    echo "$i finish"
    cd ../
done
