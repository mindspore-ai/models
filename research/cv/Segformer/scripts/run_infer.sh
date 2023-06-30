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
if [ $# != 4 ] && [ $# != 5 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_infer.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH] "
  echo "bash run_infer.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH] [OUTPUT_PATH](optional)"
  echo "For example: bash run_infer.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/checkpoint/segformer_mit_b1_10.ckpt /segformer/data/cityscapes/"
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

if [ $# == 5 ]
then
  OUTPUT_PATH=$(get_real_path $5)
fi

echo "DEVICE_ID=${DEVICE_ID}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

export PYTHONUNBUFFERED=1
export RANK_SIZE=${RANK_SIZE}
export DEVICE_NUM=${DEVICE_NUM}
export DEVICE_ID=${DEVICE_ID}

work_dir=run_infer_device$DEVICE_ID
rm -rf $work_dir
mkdir $work_dir
cp ../*.py ../src ../config ./$work_dir -r
cd ./$work_dir
echo "start infer for device $DEVICE_ID"
env > env.log

if [ $# == 4 ]
then
  python infer.py --config_path=$CONFIG_FILE --infer_ckpt_path=$CKPT_PATH --data_path=$DATASET_PATH > infer.log 2>&1 &
fi

if [ $# == 5 ]
then
  python infer.py --config_path=$CONFIG_FILE --infer_ckpt_path=$CKPT_PATH --data_path=$DATASET_PATH --infer_output_path=$OUTPUT_PATH > infer.log 2>&1 &
fi


cd ../
