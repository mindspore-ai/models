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

if [ $# -ne 1 ]
then
    echo "Usage: sh run_distribute_train_gpu_fp16.sh [DATA_PATH]"
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
echo $PATH1
if [ ! -d $PATH1 ]
then
    echo "error: IMAGE_PATH=$PATH1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train_fp16" ];
then
    rm -rf ./train_fp16
fi

rm -rf ./train_fp16
mkdir ./train_fp16
cp ../*.py ./train_fp16
cp *.sh ./train_fp16
cp ../*.yaml ./train_fp16
cp -r ../src ./train_fp16
cd ./train_fp16 || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py --data_path=$PATH1 --output_path './output' --device_target='GPU' --checkpoint_path='./' --enable_fp16_gpu=True > train.log 2>&1 &
cd ..
