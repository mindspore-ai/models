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
if [ $# -lt 3 ]
then
    echo "Usage: bash ./scripts/run_distributed_train_gpu.sh [CUDA_VISIBLE_DEVICES] [DEVICE_NUM] [DATA_PATH]"
exit 1
fi

dataset_name="imagenet"
export RANK_SIZE=$1
export DEVICE_NUM=$2
export CUDA_VISIBLE_DEVICES=$1
DATA_PATH=$3

mpirun -n ${DEVICE_NUM} --allow-run-as-root --output-filename log_output \
--merge-stderr-to-stdout python train.py \
--device_target="GPU" --dataset_name=$dataset_name \
--data_path=$DATA_PATH > log.txt 2>&1 &

