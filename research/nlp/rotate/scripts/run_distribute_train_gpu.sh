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

if [ $# != 3 ] ; then
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_gpu.sh DEVICE_NUM BATCH_SIZE MAX_STEPS"
echo "for example: bash scripts/run_distribute_train_gpu.sh 8 64 560000"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/output-distribute-gpu.log"
exit 1;
fi

export RANK_SIZE=$1
BATCH_SIZE=$2
MAX_STEPS=$3
OUTPUT_PATH=./checkpoints/rotate-distribute-gpu/

if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir -p $OUTPUT_PATH
fi

if [ ! -d "ms_log" ]; then
        mkdir ms_log
fi

mpirun --allow-run-as-root -n $RANK_SIZE \
python -u train.py \
    --device_target=GPU \
    --output_path=$OUTPUT_PATH\
    --batch_size=$BATCH_SIZE\
    --max_steps=$MAX_STEPS\
    --experiment_name=rotate-distribute-gpu\
      > ms_log/output-distribute-gpu.log 2>&1 &
