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
if [ $# != 4 ]; then
  echo "Usage: 
        bash run_eval.sh [DEVICE_TARGET] [ONNX_PATH][CKPT_PATH] [DATASET]
       " 
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

DEVICE_TARGET=$1

ONNX_PATH=$(get_real_path $2)
echo "ONNX_PATH: "$ONNX_PATH

CKPT_PATH=$(get_real_path $3)
echo "CKPT_PATH: "$CKPT_PATH

DATASET=$(get_real_path $4)
echo "DATASET_ROOT "$DATASET

if [ ! -d $DATASET ]
then
    echo "error: dataset_root=$DATASET is not a directory."
exit 1
fi

if [ ! -f $ONNX_PATH ]
then
    echo "error: ONNX_PATH=$ONNX_PATH is not a file."
exit 1
fi

if [ ! -f $CKPT_PATH ]
then
    echo "error: CKPT_PATH=$CKPT_PATH is not a file."
exit 1
fi

if [ -d "$BASE_PATH/../eval" ];
then
    rm -rf $BASE_PATH/../eval
fi
mkdir $BASE_PATH/../eval
cd $BASE_PATH/../eval || exit

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

echo "start eval"
env > env.log
echo
python $BASE_PATH/../eval_onnx.py --ONNX_device_target $DEVICE_TARGET --ONNX_CKPT_PATH $ONNX_PATH --MODEL_PRETRAINED $CKPT_PATH --DATASET_ROOT $DATASET  &> onnx_eval.log &
