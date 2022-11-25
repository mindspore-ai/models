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

usage() {
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_distribute_train_ascend.sh [DATA_ROOT] [TRAIN_ANN_FILE] [ORI_ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)"
  echo "For example: bash run_distribute_train_ascend.sh /home/coco/images/ /home/coco/annotations/train_15.json /home/coco/annotations/train.json /home/output /home/faster_rcnn-12_7393.ckpt"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
}

export GLOG_v=3
export PYTHONUNBUFFERED=1

if [ $# -lt 4 ]; then
  usage
  exit 1
fi


DATA_ROOT=$1
ANN_FILE=$2
ORI_ANN_FILE=$3
OUTPUT_DIR=$4
PRE_TRAINED=$5
DEVICE_TARGET="GPU"

FIRST_STAGE_OUTPUT=$OUTPUT_DIR/first_stage
SECOND_STAGE_OUTPUT=$OUTPUT_DIR/second_stage

mkdir -p $FIRST_STAGE_OUTPUT
mkdir -p $SECOND_STAGE_OUTPUT

echo "=============================================================================================================="
echo "train in first stage"
bash run_distribute_train_model_gpu.sh $DATA_ROOT $ANN_FILE $FIRST_STAGE_OUTPUT $PRE_TRAINED

echo "=============================================================================================================="
echo "select top 25% valuable unlabel data"
NEW_JSON=$FIRST_STAGE_OUTPUT/new_ann_file.json
bash select_valuable_sample.sh $FIRST_STAGE_OUTPUT/model_0.ckpt $ORI_ANN_FILE $ANN_FILE $NEW_JSON $DEVICE_TARGET 0

echo "=============================================================================================================="
echo "train in second stage"
bash run_distribute_train_model_gpu.sh $DATA_ROOT $NEW_JSON $SECOND_STAGE_OUTPUT $PRE_TRAINED
