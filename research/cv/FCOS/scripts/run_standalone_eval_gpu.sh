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
    echo "Usage: bash run_standalone_eval_gpu.sh [EVAL_DATA_PATH] [ANNO_DATA_PATH] [CKPT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
EVAL_PATH=$(get_real_path $1)
ANNO_PATH=$(get_real_path $2)
CKPT_PATH=$(get_real_path $3)

if [ ! -d $EVAL_PATH ]
then
    echo "error: EVAL_DATA_PATH=$EVAL_PATH is not a directory"
exit 1
fi

if [ ! -f $ANNO_PATH ]
then
    echo "error: ANNO_DATA_PATH=$ANNO_PATH is not a file"
exit 1
fi

if [ ! -f $CKPT_PATH ]
then
    echo "error: CKPT_PATH=$CKPT_PATH is not a file"
exit 1
fi


export DEVICE_ID=$4

rm -rf eval
mkdir eval
cp -r ../src/ ./eval
cp ../eval.py ./eval
echo "start eval"
cd ./eval
export EVAL_PATH=$1
export ANNO_PATH=$2
export CKPT_PATH=$3

python eval.py --device_id=$DEVICE_ID --eval_path=$EVAL_PATH  --anno_path=$ANNO_PATH --ckpt_path=$CKPT_PATH    > eval.log 2>&1 &

