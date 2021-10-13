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

if [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_train1p.sh [PROJECT_PATH] [DEVICE_ID]"
    echo "for example: bash scripts/run_train1p.sh /root/ICNet/ 0"
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


if [ ! -d $PATH1 ]
then
    echo "error: PROJECT_PATH=$PATH1 is not a directory"
exit 1
fi


rm -rf LOG
mkdir ./LOG
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$2
echo "start training for device $DEVICE_ID"
env > env.log

python3 train.py --project_path=$PATH1 > log.txt 2>&1 &
