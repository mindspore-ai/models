#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
echo "bash run_distribute_train.sh [DEVICE_ID] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]"
echo "For example:
bash run.sh 0 ucf101 \\
~/dataset/ucf101/jpg/ \\
~/dataset/ucf101/json/ucf101_01.json \\
~/results/ \\
~/pretrain_ckpt/pretrain.ckpt"
echo "It is better to use the ABSOLUTE path."
echo "=============================================================================================================="
set -e

if [ $# != 6 ]
then
  echo "Usage: bash run_standalone_train.sh [DEVICE_ID] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]"
exit 1
fi

DEVICE_ID=$1
DATASET=$2
VIDEO_PATH=$3
ANNOTATION_PATH=$4
RESULT_PATH=$5
PRETRAIN_PATH=$6

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export DEVICE_ID=$DEVICE_ID
export RANK_ID=$DEVICE_ID
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ..
env > env.log
python train.py --is_modelarts False --run_distribute False --config_path ./${DATASET}_config.yaml \
--device_id $DEVICE_ID --video_path $VIDEO_PATH --annotation_path $ANNOTATION_PATH \
--result_path $RESULT_PATH --pretrain_path $PRETRAIN_PATH > train_$DATASET.log 2>&1 &
echo "Training. Check it at train_$DATASET.log."