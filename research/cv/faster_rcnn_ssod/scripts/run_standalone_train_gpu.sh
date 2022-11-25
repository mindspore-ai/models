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
  echo "bash run_distribute_train_combine.sh [DEVICE_ID] [DATA_ROOT] [TRAIN_ANN_FILE] [ORI_ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)"
  echo "For example: bash run_standalone_train_model.sh 0 /home/coco/images/ /home/coco/annotations/train_15.json /home/coco/annotations/train.json /home/outputs /home/faster_rcnn-12_7393.ckpt"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
}

export GLOG_v=3
export PYTHONUNBUFFERED=1

if [ $# -lt 5 ]; then
  usage
  exit 1
fi

DEVICE_ID=$1
DATA_ROOT=$2
ANN_FILE=$3
ORI_ANN_FILE=$4
OUTPUT_DIR=$5
PRE_TRAINED=$6
DEVICE_TARGET="GPU"

FIRST_STAGE_OUTPUT=$OUTPUT_DIR/first_stage
SECOND_STAGE_OUTPUT=$OUTPUT_DIR/second_stage

mkdir -p $FIRST_STAGE_OUTPUT
mkdir -p $SECOND_STAGE_OUTPUT

export DEVICE_ID=$DEVICE_ID

echo "=============================================================================================================="
echo "train in first stage"
bash run_standalone_train_model.sh $DEVICE_TARGET $DEVICE_ID $DATA_ROOT $ANN_FILE $FIRST_STAGE_OUTPUT $PRE_TRAINED

echo "=============================================================================================================="
echo "select top 25% valuable unlabel data"
NEW_ANNO=$FIRST_STAGE_OUTPUT/new_ann_file.json
bash select_valuable_sample.sh $FIRST_STAGE_OUTPUT/model_0.ckpt $ORI_ANN_FILE $ANN_FILE $NEW_ANNO $DEVICE_TARGET $DEVICE_ID

echo "=============================================================================================================="
echo "train in second stage"
bash run_standalone_train_model.sh $DEVICE_TARGET $DEVICE_ID $DATA_ROOT $NEW_ANNO $SECOND_STAGE_OUTPUT $PRE_TRAINED
