#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

if [ $# != 1 ]
then
    echo "Usage: sh run_train_ascend.sh [DATASET_PATH]"
    exit 1
fi
DATASET_PATH=$1

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval

cp ../*.py ./eval
cp ../*.yaml ./eval
cp *.sh ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cd ./eval || exit
env > env.log
echo "start evaluation"

python eval.py --datapath=$DATASET_PATH --ckptpath=../ckpts &> log &

cd ..
