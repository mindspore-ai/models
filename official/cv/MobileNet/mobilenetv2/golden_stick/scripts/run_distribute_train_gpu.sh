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

CURPATH="$(dirname "$0")"

if [ $# != 4 ] && [ $# != 6 ]; then
    echo "Usage: bash run_distribute_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [DEVICE_NUM] [CKPT_TYPE](optional) [CKPT_PATH](optional)"
    echo "Examples:"
    echo "  Train from the beginning:"
    echo "    bash run_distribute_train_gpu.sh /path/to/train.py mobilenetv2_cifar10_config.yaml /path/to/dataset 2"
    echo "  Train from full precision checkpoint:"
    echo "    bash run_distribute_train_gpu.sh /path/to/train.py mobilenetv2_cifar10_config.yaml /path/to/dataset 2 FP32 /path/to/fp32_ckpt"
    echo "  Train from pretrained checkpoint:"
    echo "    bash run_distribute_train_gpu.sh /path/to/train.py mobilenetv2_cifar10_config.yaml /path/to/dataset 2 PRETRAINED /path/to/pretrained_ckpt"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PYTHON_PATH=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
DATASET_PATH=$(get_real_path $3)
DEVICE_NUM=$4

if [ ! -d $PYTHON_PATH ]
then
    echo "error: PYTHON_PATH=$PYTHON_PATH is not a directory"
    exit 1
fi

if [ ! -f $CONFIG_FILE ]
then
    echo "error: CONFIG_FILE=$CONFIG_FILE is not a file"
    exit 1
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
    exit 1
fi

if [ $# == 6 ]; then
  CKPT_TYPE=$5
  CKPT_FILE=$(get_real_path $6)

  if [ "x$CKPT_TYPE" != "xFP32" ] && [ "x$CKPT_TYPE" != "xPRETRAINED" ]; then
      echo "error: CKPT_TYPE=$CKPT_TYPE is not valid, should be FP32 or PRETRAINED"
      exit 1
  fi
  if [ ! -f $CKPT_FILE ]; then
      echo "error: CKPT_FILE=$CKPT_FILE is not a file"
      exit 1
  fi
fi

ulimit -u unlimited
export DEVICE_NUM=$4
export RANK_SIZE=$4

rm -rf ./train_parallel
mkdir ./train_parallel
cp ${PYTHON_PATH}/*.py ./train_parallel
cp -r ${CURPATH}/../../src ./train_parallel
cd ./train_parallel || exit

if [ "x$CKPT_TYPE" == "xFP32" ]; then
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --config_path=$CONFIG_FILE --run_distribute=True --platform="GPU" \
         --dataset_path=$DATASET_PATH --fp32_ckpt=$CKPT_FILE --output_path './output' \
         --device_num=$DEVICE_NUM &> log &
elif [ "x$CKPT_TYPE" == "xPRETRAINED" ]; then
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --config_path=$CONFIG_FILE --run_distribute=False --platform="GPU" \
         --dataset_path=$DATASET_PATH --pretrain_ckpt=$CKPT_FILE --output_path './output' \
         --device_num=$DEVICE_NUM &> log &
else
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python train.py --config_path=$CONFIG_FILE --run_distribute=False --platform="GPU" \
         --dataset_path=$DATASET_PATH --output_path './output' --device_num=$DEVICE_NUM &> log &
fi
