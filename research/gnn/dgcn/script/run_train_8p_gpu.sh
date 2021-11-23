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

if [ $# != 3 ]
then 
    echo "Usage: bash run_train_8p_gpu.sh [DATASET_NAME] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]"
exit 1
fi

DATASET_NAME=$1
echo $DATASET_NAME

DEVICE_NUM=$2
echo $DEVICE_NUM

export DEVICE_NUM=$2
export RANK_SIZE=$DEVICE_NUM
export CUDA_VISIBLE_DEVICES="$3"

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train
cp -r ../data ./train
cd ./train || exit
env > env.log
echo "start training on device $3"

if [ $DATASET_NAME == cora ] || [ $DATASET_NAME == citeseer ] || [ $DATASET_NAME == pubmed ]
then
    nohup mpirun -n $DEVICE_NUM --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python -u train.py --dataset=$DATASET_NAME --device_target=GPU --distributed=True > train_gpu.log 2>&1 &
fi

cd ..
