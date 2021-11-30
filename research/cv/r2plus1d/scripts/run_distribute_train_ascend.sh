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

DIR="$( cd "$( dirname "$0"  )" && pwd  )"

# help message
if [ $# != 3 ]; then
  echo "Usage: bash run_distribute_train_ascend.sh [rank_size] [rank_start_id] [rank_table_file]"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export SLOG_PRINT_TO_STDOUT=0
export RANK_TABLE_FILE=$3
export RANK_SIZE=$1
export RANK_START_ID=$2

rm -rf $DIR/../ascend_work_space
mkdir $DIR/../ascend_work_space

for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    if [ -d $DIR/../ascend_work_space/device${DEVICE_ID} ]; then
      rm -rf $DIR/../ascend_work_space/device${DEVICE_ID}
    fi
    mkdir $DIR/../ascend_work_space/device${DEVICE_ID}
    cp -r $DIR/../src $DIR/../ascend_work_space/device${DEVICE_ID}
    cp $DIR/../train.py $DIR/../default_config.yaml $DIR/../ascend_work_space/device${DEVICE_ID}
    cd $DIR/../ascend_work_space/device${DEVICE_ID} || exit
    nohup python ./train.py --device_target=Ascend --is_distributed=1 > log.txt 2>&1 &
done
