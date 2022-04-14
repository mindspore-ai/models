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
echo "bash run_distribute_train_gpu.sh RANK_SIZE"
echo "For example: bash run_distribute_train_gpu.sh 8"
echo "=============================================================================================================="
set -e

export DEVICE_NUM=$1
export RANK_SIZE=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "start training"
mpirun -n $1 --allow-run-as-root --output-filename log_output1 --merge-stderr-to-stdout \
    python ./train.py --device_num $1  --device_target "GPU" --run_distribute True \
    > train_dis.log 2>&1 &

