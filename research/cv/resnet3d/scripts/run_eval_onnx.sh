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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_onnx.sh [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [ONNX_PATH]"
echo "For example:
bash run_eval_onnx.sh 0 ucf101 \\
/path/ucf101/jpg/ \\
/path/ucf101/json/ucf101_01.json \\
/path/resnet-3d.onnx"
echo "It is better to use the ABSOLUTE path."
echo "=============================================================================================================="
set -e

if [ $# != 4 ]
then
  echo "Usage: bash run_eval_onnx.sh [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [ONNX_PATH]"
exit 1
fi

DATASET=$1
VIDEO_PATH=$2
ANNOTATION_PATH=$3
ONNX_PATH=$4

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
cd ..
env > env.log
echo "Eval begin"
python eval_onnx.py --is_modelarts False  --config_path ./${DATASET}_config.yaml --video_path $VIDEO_PATH \
--annotation_path $ANNOTATION_PATH --onnx_path $ONNX_PATH --device_target GPU > eval_$DATASET.log 2>&1 &

echo "Evaling. Check it at eval_$DATASET.log"

