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
  echo "bash run_distribute_train_combine.sh [PRETRAINED] [ORI_ANN_FILE] [PICKED_ANN_FILE] [ANN_FILE] [VALUABLE_ANN_FILE] [DEVICE_TARGET] [DEVICE_ID]"
  echo "For example: bash run_standalone_train_model.sh /home/faster_rcnn-12_7393.ckpt /home/coco/annotations/train.json /home/coco/annotations/train_25.json /home/coco/annotations/train_60.json Ascend 0"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
}

if [ $# -ne 6 ]; then
  usage
  exit 1
fi

PRETRAINED=$1
ORI_ANN_FILE=$2
PICKED_ANN_FILE=$3
VALUABLE_ANN_FILE=$4
DEVICE_TARGET=$5

export DEVICE_ID=$6

cd ..
python infer.py --checkpoint_path=$PRETRAINED \
                --device_target=$DEVICE_TARGET \
                --eval_output_dir="./"

INFER_JSON="./infer_results.json"
python sorted_values.py --infer_json=${INFER_JSON}

TOP_VALUE_FILE="./top_value_data.json"
python pick_select.py --ann_file=$ORI_ANN_FILE \
                      --combine_ann_file=$PICKED_ANN_FILE \
                      --pick_ratio=60 \
                      --top_value_file=$TOP_VALUE_FILE \
                      --output_ann_file=$VALUABLE_ANN_FILE
