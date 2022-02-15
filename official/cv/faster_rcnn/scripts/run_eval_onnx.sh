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

if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_eval_onnx.sh [ANNO_PATH] \
[ONNX_MODEL] [BACKBONE] [COCO_ROOT] \
[MINDRECORD_DIR]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ANNO_PATH=$(get_real_path $1)
ONNX_MODEL=$(get_real_path $2)
BACKBONE=$3
COCO_ROOT=$(get_real_path $4)
MINDRECORD_DIR=$(get_real_path $5)

if [ ! -f $ANNO_PATH ]
then
    echo "error: ANNO_PATH=$ANNO_PATH is not a file"
exit 1
fi

if [ ! -f $ONNX_MODEL ]
then
    echo "error: CHECKPOINT_PATH=$ONNX_MODEL is not a file"
exit 1
fi

if [ ! -d $COCO_ROOT ]
then
    echo "error: COCO_ROOT=$COCO_ROOT is not a dir"
exit 1
fi

if [ ! -d $MINDRECORD_DIR ]
then
    echo "error: mindrecord_dir=$MINDRECORD_DIR is not a dir"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $BACKBONE == 'resnet_v1.5_50' ]; then
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
elif [ $BACKBONE == 'resnet_v1_101' ]; then
  CONFIG_FILE="${BASE_PATH}/../default_config_101.yaml"
elif [ $BACKBONE == 'resnet_v1_152' ]; then
  CONFIG_FILE="${BASE_PATH}/../default_config_152.yaml"
elif [ $BACKBONE == 'resnet_v1_50' ]; then
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
elif [ $BACKBONE == 'inception_resnet_v2' ]; then
  CONFIG_FILE="${BASE_PATH}/../default_config_InceptionResnetV2.yaml"
else
  echo "Unrecognized parameter"
  exit 1
fi

python eval_onnx.py \
  --config_path=$CONFIG_FILE \
  --coco_root=$COCO_ROOT \
  --mindrecord_dir=$MINDRECORD_DIR \
  --device_target="GPU" \
  --anno_path=$ANNO_PATH \
  --file_name=$ONNX_MODEL \
  --backbone=$BACKBONE &> log &
