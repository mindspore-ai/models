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

if [ $# != 5 ] ; then
echo "Please run the script as: "
echo "bash scripts/run_standalone_train.sh DEVICE_ID DEVICE_TARGET OUTPUT_PATH MAX_STEPS LOG_FILE"
echo "for example: bash scripts/run_standalone_train.sh 0 Ascend ./checkpoints/rotate-standalone-ascend/ 80000 output-standalone-ascend.log"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/output-standalone-ascend.log"
exit 1;
fi

DEVICE_TARGET=$2

if [ "$DEVICE_TARGET" = "GPU" ]
then
  export CUDA_VISIBLE_DEVICES=$1
fi

if [ "$DEVICE_TARGET" = "Ascend" ];
then
  export DEVICE_ID=$1
fi

OUTPUT_PATH=$3
MAX_STEPS=$4
LOG_FILE=$5

if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir -p $OUTPUT_PATH
fi

if [ ! -d "ms_log" ]; then
        mkdir ms_log
fi

python -u train.py \
    --device_target=$DEVICE_TARGET \
    --output_path=$OUTPUT_PATH\
    --max_steps=$MAX_STEPS\
        > ms_log/$LOG_FILE 2>&1 &

