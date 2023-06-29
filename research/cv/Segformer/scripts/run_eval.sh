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
if [ $# != 4 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_eval.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH] "
  echo "For example: bash run_eval.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/checkpoint/segformer_mit_b1_10.ckpt /segformer/data/cityscapes/ "
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
CONFIG_FILE=$(get_real_path $2)
CKPT_PATH=$(get_real_path $3)
DATASET_PATH=$(get_real_path $4)

RANK_SIZE=${DEVICE_NUM}

export PYTHONUNBUFFERED=1
echo "DEVICE_ID=${DEVICE_ID}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"

export RANK_SIZE=${RANK_SIZE}
export DEVICE_NUM=${DEVICE_NUM}
export DEVICE_ID=${DEVICE_ID}

work_dir=run_eval_device$DEVICE_ID
rm -rf $work_dir
mkdir $work_dir
cp ../*.py ../src ../config ./$work_dir -r
cd ./$work_dir
echo "start eval for device $DEVICE_ID"
env > env.log
python eval.py --config_path=$CONFIG_FILE --eval_ckpt_path=$CKPT_PATH --data_path=$DATASET_PATH > eval.log 2>&1 &
cd ../
