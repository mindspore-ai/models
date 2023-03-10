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

if [ $# != 4 ]
then 
    echo "Usage: bash script/run_eval_onnx_gpu.sh [NETWORK_NAME] [DATASET_NAME] [DATA_PATH] [ONNX_PATH]"
    echo "[NETWORK_NAME]: resnet18, resnet34, resnet50,resnet101, resnet152, seresnet50"
    echo "[DATASET_NAME]: imagenet2012, cifar10"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

NETWORK=$1
DATASET=$2
PATH1=$(get_real_path $3)
PATH2=$(get_real_path $4)

if [ ! -d $PATH1 ]
then 
    echo "error: DATA_PATH=$PATH1 is not a directory"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: ONNX_PATH=$PATH2 is not a file"
exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

echo "start evaluation"

python eval_onnx.py \
    --net_name=$NETWORK \
    --dataset=$DATASET \
    --dataset_path=$PATH1 \
    --device_target="GPU" \
    --onnx_path=$PATH2  > eval_onnx.log 2>&1 &
