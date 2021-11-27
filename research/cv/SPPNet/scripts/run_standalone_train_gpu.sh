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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [DEVICE_ID] [TRAIN_MODEL]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: TRAIN_DATA_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: EVAL_DATA_PATH=$PATH2 is not a directory"
exit 1
fi

rm -rf ./$TRAIN_MODEL
mkdir ./$TRAIN_MODEL
cp -r ../src ./$TRAIN_MODEL
cp ../train.py ./$TRAIN_MODEL
echo "start training for $TRAIN_MODEL"
cd ./$TRAIN_MODEL ||exit
python train.py --device_id=$DEVICE_ID --device_target=GPU --train_path=$TRAIN_PATH --eval_path=$EVAL_PATH  --train_model=$TRAIN_MODEL > log 2>&1 &
