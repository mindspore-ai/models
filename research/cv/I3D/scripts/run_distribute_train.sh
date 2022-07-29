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

DIR="$(cd "$(dirname "$0")" && pwd)"

# help message
if [ $# != 8 ]; then
  echo "Usage: bash ./scripts/run_distribute_train.sh [rank_size] [rank_table_file] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path](necessary)"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2

rm -rf $DIR/output_distribute
mkdir $DIR/output_distribute

for ((i = 0; i < ${RANK_SIZE}; i++)); do
  export RANK_ID=${i}
  export DEVICE_ID=${i}
  echo "start training for device $i"
  if [ -d $DIR/output_distribute/$3_$4_device${i} ]; then
    rm -rf $DIR/output_distribute/$3_$4_device${i}
  fi
  mkdir $DIR/output_distribute/$3_$4_device${i}
  nohup python ./train.py \
    --device_id=${i} \
    --dataset=$3 \
    --video_path=$6 \
    --annotation_path=$7 \
    --checkpoint_path=$8 \
    --mode=$4 \
    --num_epochs=$5 \
    --distributed=True >$DIR/output_distribute/$3_$4_device${i}_log.txt 2>&1 &
done
