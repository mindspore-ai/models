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

if [[ $# != 2 && $# != 3 ]]
then 
    echo "Usage: bash run_train.sh [DATASET_PATH] [DATASET_NAME] [DEVICE_ID](optional)"
exit 1
fi

DATASET_PATH=$1
DATASET_NAME=$2
echo $DATASET_NAME

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
if [ -z $3 ]
then
    export DEVICE_ID=0
else
    export DEVICE_ID=$3
fi
export RANK_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp *.sh ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cd ./train || exit
env > env.log
echo "start training for device $DEVICE_ID"


if [ $DATASET_NAME == cora ]
then
    python train.py --data_dir=$DATASET_PATH --train_nodes_num=140 &> log &
fi

if [ $DATASET_NAME == citeseer ]
then
    python train.py --data_dir=$DATASET_PATH --train_nodes_num=120 &> log &
fi
cd ..

