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

if [ $# != 2 ]
then
    echo "Usage: sh run_eval_gpu.sh [DATASET_PATH] [ONNX_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATASET_PATH=$(get_real_path $1)
CHECKPOINT_PATH=$(get_real_path $2)
echo $DATASET_PATH
echo $CHECKPOINT_PATH



export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval_onnx" ];
then
    rm -rf ./eval_onnx
fi
mkdir ./eval_onnx
cp ../*.py ./eval_onnx
cp ../*.yaml ./eval_onnx
cp -r ../src ./eval_onnx
cp -r ../model_utils ./eval_onnx
cd ./eval_onnx || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval_onnx.py \
    --device_target="GPU" \
    --data_dir=$DATASET_PATH \
    --pretrained=$CHECKPOINT_PATH \
    --testing_shape=416 > log.txt 2>&1 &
cd ..
