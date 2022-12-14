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

if [ $# != 2 ] && [ $# != 3 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_standalone_train_gpu.sh DEVICE_ID MINDRECORD_DIR LOAD_CHECKPOINT_PATH(optional)"
  echo "for example: bash run_standalone_train_gpu.sh 0 /path/mindrecord_dataset /path/load_ckpt"
  echo "if no ckpt, just run: bash run_standalone_train_gpu.sh 0 /path/mindrecord_dataset"
  echo "=============================================================================================================="
exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1
MINDRECORD_DIR=$2
if [ $# == 3 ];
then
    LOAD_CHECKPOINT_PATH=$3
else
    LOAD_CHECKPOINT_PATH=""
fi

mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export DEVICE_ID=$DEVICE_ID

CONFIG=$(get_real_path "$PROJECT_DIR/../centernetdet_gpu_config.yaml")

python ${PROJECT_DIR}/../train.py  \
    --config_path $CONFIG \
    --distribute=false \
    --need_profiler=false \
    --device_id=$DEVICE_ID \
    --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
    --save_checkpoint_num=1 \
    --mindrecord_dir=$MINDRECORD_DIR \
    --mindrecord_prefix="coco_det.train.mind" \
    --visual_image=false \
    --save_result_dir="" > training_log.txt 2>&1 &
