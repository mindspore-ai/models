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
if [ $# != 4 ]
then
    echo "Usage:
          bash run_eval_cpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [CHECKPOINT_PATH]
          "
exit 1
fi

# check dataset type
if [[ $1 != "ImageNet" ]] && [[ $1 != "CIFAR10" ]]
then
    echo "error: Only supported for ImageNet and CIFAR10, but DATASET_TYPE=$1."
exit 1
fi

# check dataset file
if [ ! -d $2 ]
then
    echo "error: DATASET_PATH=$2 is not a directory."
exit 1
fi

# check model name
if [[ $3 != "efficientnet_b0" ]] && [[ $3 != "efficientnet_b1" ]]
then
    echo "error: Only supported for efficientnet_b0 and efficientnet_b1, but MODEL_NAME=$3."
exit 1
fi

# check checkpoint file
if [ ! -f $4 ]
then
    echo "error: CHECKPOINT_PATH=$4 is not a file"
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

if [[ $3 == "efficientnet_b0" ]] && [[ $1 == "ImageNet" ]]; then
    CONFIG_FILE="${BASEPATH}/../efficientnet_b0_imagenet_config.yaml"
elif [[ $3 == "efficientnet_b0" ]] && [[ $1 == "CIFAR10" ]]; then
    CONFIG_FILE="${BASEPATH}/../efficientnet_b0_cifar10_config.yaml"
elif [[ $3 == "efficientnet_b1" ]] && [[ $1 == "ImageNet" ]]; then
    CONFIG_FILE="${BASEPATH}/../efficientnet_b1_imagenet_config.yaml"
else
    echo "Unrecognized parameter"
    exit 1
fi


if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

python ${BASEPATH}/../eval.py --config_path $CONFIG_FILE --dataset $1 --data_path $2 --platform CPU --model $3 --checkpoint=$4 > ./eval.log 2>&1 &
