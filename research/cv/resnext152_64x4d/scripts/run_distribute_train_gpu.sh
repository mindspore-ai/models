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

if [ $# != 1 ]
then
    echo "===================================================================================================="
    echo "Please run the script as: "
    echo "bash run_distribute_train_gpu.sh DATA_DIR"
    echo "for example: bash run_distribute_train_gpu.sh /path/ImageNet2012/train"
    echo "===================================================================================================="
    exit 1
fi

DATA_DIR=$1

rm -rf logs
mkdir ./logs

echo "Start OpenMPI"

mpirun -n 8 --allow-run-as-root python ./train.py \
    --is_distributed=1 \
    --platform=GPU \
    --per_batch_size=16 \
    --lr=0.05 \
    --data_dir=$DATA_DIR \
    --use_python_multiprocessing=1 \
    --data_sink_mode=1 \
    --run_eval=0 > ./logs/log.txt 2>&1 &
