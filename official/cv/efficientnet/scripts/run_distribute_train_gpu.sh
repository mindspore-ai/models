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
if [ $# != 5 ] && [ $# != 6 ]
then
    echo "Usage:
          bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [PRETRAINED_CKPT_PATH](optional)
          "
exit 1
fi

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi

# check dataset type
if [[ $3 != "ImageNet" ]] && [[ $3 != "CIFAR10" ]]
then
    echo "error: Only supported for ImageNet and CIFAR10, but DATASET_TYPE=$3."
exit 1
fi

# check dataset file
if [ ! -d $4 ]
then
    echo "error: DATASET_PATH=$4 is not a directory"
exit 1
fi

# check model name
if [[ $5 != "efficientnet_b0" ]] && [[ $5 != "efficientnet_b1" ]]
then
    echo "error: Only supported for efficientnet_b0 and efficientnet_b1, but MODEL_NAME=$5."
exit 1
fi

export DEVICE_NUM=$1
export RANK_SIZE=$1

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

if [[ $5 == "efficientnet_b0" ]] && [[ $3 == "ImageNet" ]]; then
    CONFIG_FILE="${BASEPATH}/../efficientnet_b0_imagenet_config.yaml"
elif [[ $5 == "efficientnet_b0" ]] && [[ $3 == "CIFAR10" ]]; then
    CONFIG_FILE="${BASEPATH}/../efficientnet_b0_cifar10_config.yaml"
elif [[ $5 == "efficientnet_b1" ]] && [[ $3 == "ImageNet" ]]; then
    CONFIG_FILE="${BASEPATH}/../efficientnet_b1_imagenet_config.yaml"
else
    echo "Unrecognized parameter"
    exit 1
fi

if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

export CUDA_VISIBLE_DEVICES="$2"

if [ $# == 5 ]
then
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ${BASEPATH}/../train.py \
        --config_path $CONFIG_FILE \
        --distributed True \
        --platform GPU \
        --dataset $3 \
        --data_path $4 \
        --model $5 > train.log 2>&1 &
fi

if [ $# == 6 ]
then
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ${BASEPATH}/../train.py \
        --config_path $CONFIG_FILE \
        --platform GPU \
        --distributed True \
        --dataset $3 \
        --data_path $4 \
        --model $5 \
        --resume $6 > train.log 2>&1 &
fi

