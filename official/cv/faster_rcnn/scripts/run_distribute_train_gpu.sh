#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh DEVICE_NUM PRETRAINED_PATH BACKBONE COCO_ROOT MINDRECORD_DIR(option)"
echo "for example: bash run_distribute_train_gpu.sh 8 /path/pretrain.ckpt resnet_v1_50 cocodataset mindrecord_dir(option)"
echo "It is better to use absolute path."
echo "=============================================================================================================="

if [ $# -le 3 ]
then
    echo "Usage: 
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
    "
exit 1
fi

if [ $3 != "resnet_v1_50" ] && [ $3 != "resnet_v1.5_50" ] && [ $3 != "resnet_v1_101" ] && [ $3 != "resnet_v1_152" ] && [ $3 != "inception_resnet_v2" ]
then 
  echo "error: the selected backbone must be resnet_v1_50, resnet_v1.5_50, resnet_v1_101, resnet_v1_152, inception_resnet_v2"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export RANK_SIZE=$1
PRETRAINED_PATH=$2
PATH3=$4

mindrecord_dir=$PATH3/MindRecord_COCO_TRAIN/
if [ $# -eq 5 ]
then
    mindrecord_dir=$(get_real_path $5)
    if [ ! -d $mindrecord_dir ]
    then
        echo "error: mindrecord_dir=$mindrecord_dir is not a dir"
    exit 1
    fi
fi
echo $mindrecord_dir

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $# -ge 1 ]; then
  if [ $3 == 'resnet_v1.5_50' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $3 == 'resnet_v1_101' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_101.yaml"
  elif [ $3 == 'resnet_v1_152' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_152.yaml"
  elif [ $3 == 'resnet_v1_50' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $3 == 'inception_resnet_v2' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_InceptionResnetV2.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/default_config.yaml"
fi

if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit

echo "start training on $RANK_SIZE devices"
env > env.log
mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
    python ${BASE_PATH}/../train.py  \
    --config_path=$CONFIG_FILE \
    --run_distribute=True \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --pre_trained=$PRETRAINED_PATH \
    --backbone=$3 \
    --coco_root=$PATH3 \
    --mindrecord_dir=$mindrecord_dir > train.log 2>&1 &