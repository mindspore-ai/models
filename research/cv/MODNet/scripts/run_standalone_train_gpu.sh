#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [DEVICE_ID]"
    echo "for example: bash scripts/run_standalone_train_gpu.sh /path/to/config.yml 0"
    echo "=============================================================================================================="
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



if [ ! -f $PATH1 ]
then
    echo "error: PROJECT_PATH=$PATH1 is not a directory"
exit 1
fi

export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$2
echo "start training for device $DEVICE_ID"
env > env.log

if [ -d "train" ]; then
    rm -rf ./train
fi
mkdir ./train
cp ./*.py ./train
cp -r ./src ./train
cp -r ./pretrained ./train
cd ./train || exit


nohup python train.py --config=$PATH1 --device_target='GPU' --device_id=$DEVICE_ID --output_path='./output' > train.log 2>&1 &

cd ..
