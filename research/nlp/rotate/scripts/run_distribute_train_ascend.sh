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

if [ $# != 4 ] ; then
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_ascend.sh DEVICE_NUM BATCH_SIZE MAX_STEPS RANK_TABLE_FILE"
echo "for example: bash scripts/run_distribute_train_ascend.sh 8 64 640000 ./rank_table_8p.json"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/output-distribute-ascend.log"
exit 1;
fi

export RANK_SIZE=$1
BATCH_SIZE=$2
MAX_STEPS=$3
export RANK_TABLE_FILE=$4
OUTPUT_PATH=./checkpoints/rotate-distribute-ascend/

if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir -p $OUTPUT_PATH
fi

if [ ! -d "ms_log" ]; then
        mkdir ms_log
fi

for ((i=0;i<$RANK_SIZE;i++))
do
  export DEVICE_ID=$i
  export RANK_ID=$i
  python -u train.py \
      --device_target=Ascend \
      --output_path=$OUTPUT_PATH \
      --batch_size=$BATCH_SIZE\
      --max_steps=$MAX_STEPS\
      --experiment_name=rotate-distribute-ascend\
        > ms_log/output-distribute-ascend.log 2>&1 &
done