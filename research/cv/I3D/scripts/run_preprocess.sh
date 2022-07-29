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
  echo "Usage: bash ./scripts/run_preprocess.sh [rank_size] [rank_table_file] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path]"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export SLOG_PRINT_TO_STDOUT=0
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2

for ((i = 0; i <= $RANK_SIZE - 1; i++)); do
  export RANK_ID=${i}
  export DEVICE_ID=$((i + RANK_START_ID))
  echo 'start rank='${i}', device id='${DEVICE_ID}'...'
  if [ -d $DIR/output_preprocess/$4/$5/$4_$5_device${DEVICE_ID} ]; then
    rm -rf $DIR/output_preprocess/$4/$5/$4_$5_device${DEVICE_ID}
  fi
  mkdir $DIR/output_preprocess/$3/
  mkdir $DIR/output_preprocess/$3/$4/
  mkdir $DIR/output_preprocess/$3/$4/$3_$4_device${DEVICE_ID}
  touch $DIR/output_preprocess/$3/$4/$3_$4_device${DEVICE_ID}_log.txt
  nohup python ./preprocess.py \
    --sink_mode=False \
    --device_id=${DEVICE_ID} \
    --dataset=$3 \
    --video_path=$6 \
    --annotation_path=$7 \
    --checkpoint_path=$8 \
    --mode=$4 \
    --num_epochs=$5 \
    --distributed=True >$DIR/output_preprocess/$3/$4/$3_$4_device${DEVICE_ID}_log.txt 2>&1 &
done
