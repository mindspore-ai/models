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
if [ $# != 4 ]; then
  echo "Usage: bash scripts/run_distribute_train.sh [rank_size] [rank_start_id] [rank_table_file] [bert_ckpt]"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export SLOG_PRINT_TO_STDOUT=0
export RANK_SIZE=$1
export RANK_START_ID=$2
export RANK_TABLE_FILE=$3
export BERT_CKPT=$4

rm -rf $DIR/output_distribute
mkdir $DIR/output_distribute

for ((i = 0; i <= $RANK_SIZE - 1; i++)); do
  export RANK_ID=${i}
  export DEVICE_ID=$((i + RANK_START_ID))
  echo 'start rank='${i}', device id='${DEVICE_ID}'...'
  if [ -d $DIR/output_distribute/device${DEVICE_ID} ]; then
    rm -rf $DIR/output_distribute/device${DEVICE_ID}
  fi
  mkdir $DIR/output_distribute/device${DEVICE_ID}

  nohup python train.py \
    --device_id ${DEVICE_ID} --bert_ckpt ${BERT_CKPT} --rank_size ${RANK_SIZE} >$DIR/output_distribute/device${DEVICE_ID}_log.txt 2>&1 &
done
