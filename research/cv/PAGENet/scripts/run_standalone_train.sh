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
    echo "Usage: bash scripts/run_standalone_train.sh [DEVICE_ID] [CONFIG_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export DEVICE_ID=$1
export CUDA_VISIBLE_DEVICES=$1
export RANK_ID=0
CONFIG_PATH=$(get_real_path $2)

if [ ! -f ${CONFIG_PATH} ]; then
echo "config path ${CONFIG_PATH} file not exists"
exit
fi
# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

cd $BASE_PATH/..

python train.py --config_path=$CONFIG_PATH &> standalone_train.log 2>&1 &

echo "The train log is at $BASE_PATH/../standalone_train.log."
