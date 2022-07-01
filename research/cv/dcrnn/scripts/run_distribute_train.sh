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

cd ..
DIR="$(cd "$(dirname "$0")" && pwd)"

# help message
if [ $# != 4 ]; then
  echo "Usage: bash run_distribute_train.sh [context] [rank_table_file] [save_dir] [distributed]"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export SLOG_PRINT_TO_STDOUT=0
export RANK_SIZE=8
export RANK_START_ID=0
export RANK_TABLE_FILE=$2

rm -rf "${DIR:?}/${3:?}/"
mkdir $DIR/$3

for ((i = 0; i <= 7; i++)); do
  export RANK_ID=${i}
  export DEVICE_ID=$((i + 0))
  echo 'start rank='${i}', device id='${DEVICE_ID}'...'
  if [ -d $DIR/output_distribute/dcrnn_device${DEVICE_ID} ]; then
    rm -rf $DIR/output_distribute/dcrnn_device${DEVICE_ID}
  fi
  nohup python ./train.py \
  --context=$1 \
  --device_id=${DEVICE_ID} \
  --save_dir=${3}\
  --distributed=${4}>$DIR/$3/dcrnn_device${DEVICE_ID}_log.txt 2>&1 &
done
