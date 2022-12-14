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

if [ $# != 1 ] && [ $# != 2 ]
then
    echo "bash run_standalone_train.sh  [DATASET_PATH]  "
    echo "bash run_standalone_train.sh  [DATASET_PATH] [PRETRAINED_CKPT_PATH] "
    exit 1
fi
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
  echo "$1"
    else
  echo "$(realpath -m $PWD/$1)"
    fi
}
ulimit -u unlimited
PATH1=$(get_real_path $1)
if [ $# == 2 ]
then 
    PATH2=$(get_real_path $2)
    fi
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp -r ./*.py ./train
cp -r ./*.yaml ./train
cp -r ./src ./train
cd ./train ||exit
if [ $# == 1 ]
then
python ./trainer_hmr.py  --data_path=$PATH1 > train.log 2>&1 &
fi
if [ $# == 2 ]
then
python ./trainer_hmr.py  --data_path=$PATH1 --checkpoint_file_path=$PATH2 > train.log 2>&1 &
fi