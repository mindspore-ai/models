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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_for_gpu.sh [DATA_PATH] [DEVICE_NUM]"
echo "for example: bash run_standalone_train_gpu.sh /path/dataset 8"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# -ne 2 ]
then
    echo "Usage: bash run_standalone_train_for_gpu.sh [DATA_PATH] [DEVICE_NUM]"
exit 1
fi

# check dataset path
if [ ! -d $1 ]
then
    echo "error: DATA_PATH=$1 is not a directory"    
exit 1
fi

export DEVICE_NUM=$2
export DATA_PATH=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "start training"
mpirun -n $DEVICE_NUM --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python train.py \
    --device_num $DEVICE_NUM \
    --data_url $DATA_PATH \
    --run_distribute True \
    > train.log 2>&1 &