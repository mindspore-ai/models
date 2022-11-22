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

if [ $# -lt 3 ]
then
    usage="Usage: bash ./infer_ssd_mobilenet_v1_fpn_onnx.sh \
<DATA_PATH> <COCO_ROOT> <ONNX_MODEL_PATH> \
[<INSTANCES_SET>] [<DEVICE_TARGET>] [<CONFIG_PATH>]"
    echo "$usage"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$1
COCO_ROOT=$2
ONNX_MODEL_PATH=$3
INSTANCES_SET=${4:-'annotations/instances_{}.json'}
DEVICE_TARGET=${5:-"GPU"}
CONFIG_PATH=${6:-"config/ssd_mobilenet_v1_fpn_ONNX_config.yaml"}

python ../infer_ssd_mobilenet_v1_fpn_onnx.py \
    --dataset coco \
    --data_path $DATA_PATH \
    --coco_root $COCO_ROOT \
    --instances_set $INSTANCES_SET \
    --file_name $ONNX_MODEL_PATH \
    --device_target $DEVICE_TARGET \
    --config_path $CONFIG_PATH \
    --batch_size 1 \
    &> eval.log &
