#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
if [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ] && [ $# != 5 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE]"
  echo "bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional)"
  echo "bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional)"
  echo "bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional) [RUN_EVAL](optional)"
  echo "For example: bash run_standalone_train.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/data/cityscapes/"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1
DEVICE_NUM=1
RANK_SIZE=${DEVICE_NUM}
CONFIG_FILE=$(get_real_path $2)

if [ $# == 3 ]
then
  DATASET_PATH=$(get_real_path $3)
fi

if [ $# == 4 ]
then
  DATASET_PATH=$(get_real_path $3)
  PRETRAINED_CKPT_PATH=$(get_real_path $4)
fi

if [ $# == 5 ]
then
  DATASET_PATH=$(get_real_path $3)
  PRETRAINED_CKPT_PATH=$(get_real_path $4)
  RUN_EVAL=$5
fi

echo "DEVICE_ID=${DEVICE_ID}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "PRETRAINED_CKPT_PATH=${PRETRAINED_CKPT_PATH}"
echo "RUN_EVAL=${RUN_EVAL}"

export PYTHONUNBUFFERED=1
export RANK_SIZE=${RANK_SIZE}
export DEVICE_NUM=${DEVICE_NUM}
export DEVICE_ID=${DEVICE_ID}

work_dir=run_standalone_train_device$DEVICE_ID
rm -rf $work_dir
mkdir $work_dir

if [ $# == 2 ]
then
  cp ../*.py ../src ../config ../data ../pretrained ./$work_dir -r
  cd ./$work_dir
  echo "start standalone training for device $DEVICE_ID"
  env > env.log
  python train.py --config_path=$CONFIG_FILE --run_distribute=False > train.log 2>&1 &
fi

if [ $# == 3 ]
then
  cp ../*.py ../src ../config ../pretrained ./$work_dir -r
  cd ./$work_dir
  echo "start standalone training for device $DEVICE_ID"
  env > env.log
  python train.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --run_distribute=False > train.log 2>&1 &
fi

if [ $# == 4 ]
then
  cp ../*.py ../src ../config ./$work_dir -r
  cd ./$work_dir
  echo "start standalone training for device $DEVICE_ID"
  env > env.log
  python train.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --pretrained_ckpt_path=$PRETRAINED_CKPT_PATH --run_distribute=False > train.log 2>&1 &
fi

if [ $# == 5 ]
then
  cp ../*.py ../src ../config ./$work_dir -r
  cd ./$work_dir
  echo "start standalone training for device $DEVICE_ID"
  env > env.log
  python train.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --pretrained_ckpt_path=$PRETRAINED_CKPT_PATH --run_eval=$RUN_EVAL --run_distribute=False --batch_size=2 > train.log 2>&1 &
fi

cd ../
