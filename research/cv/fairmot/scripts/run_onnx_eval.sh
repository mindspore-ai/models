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
        bash run_onnx_eval.sh [device] [config] [load_ckpt] [dataset_dir]
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

CONFIG=$(get_real_path $2)
echo "CONFIG: "$CONFIG

LOAD_MODEL=$(get_real_path $3)
echo "PRETRAINED_MODEL: "$LOAD_MODEL

DATASET_DIR=$(get_real_path $4)
echo "DATASET_DIR: "$DATASET_DIR

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -f $LOAD_MODEL ]
then
    echo "error: ckpt=$LOAD_MODEL is not a file."
exit 1
fi

if [ ! -d $DATASET_DIR ]
then
    echo "error: dataset=$DATASET_DIR is not a directory."
exit 1
fi

if [ -d "$BASE_PATH/../eval" ];
then
    rm -rf $BASE_PATH/../eval
fi
mkdir $BASE_PATH/../eval
cd $BASE_PATH/../eval || exit

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

echo "start eval on device $1"
env > env.log
echo
python -u $BASE_PATH/../eval_onnx.py  --device $1 --load_model $LOAD_MODEL --data_dir $DATASET_DIR &> eval.log &
