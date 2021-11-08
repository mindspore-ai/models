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

# help message
if [ $# != 4 ]; then
  echo "Usage: sh run_train.sh [root path of code] [rank size]" \
       "[rank start id] [rank table file]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)/"
  fi
}

train_code_path=$(get_real_path $1)
echo "train_code_path: "$train_code_path

if [ ! -d $train_code_path ]
then
    echo "error: train_code_path=$train_code_path is not a dictionary."
exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export SLOG_PRINT_TO_STDOUT=0
export RANK_TABLE_FILE=$4
export RANK_SIZE=$2
export RANK_START_ID=$3

for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    if [ -d ${train_code_path}/device${DEVICE_ID} ]; then
      rm -rf ${train_code_path}/device${DEVICE_ID}
    fi
    mkdir ${train_code_path}/device${DEVICE_ID}
    cd ${train_code_path}/device${DEVICE_ID} || exit
    nohup python ${train_code_path}train.py > log.txt 2>&1 &
done