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
echo "bash run.sh [DATA_DIR] [DATA_LIST] [RESTORE_FROM]"
echo "For example: bash run.sh /path/dataset /path/datalist /path/ckpt "
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

export DATA_DIR=$1
export DATA_LIST=$2
export RESTORE_FROM=$3

mkdir ./dgc_832_8p
echo "start training"
    mpirun -output-filename mpilog -merge-stderr-to-stdout -n 8 --allow-run-as-root \
    python train.py --data_set cityscapes \
    --data_dir $1 \
    --data_list $2 \
    --restore_from $3 \
    --input_size 832 \
    --batch_size 1 \
    --learning_rate 0.01 \
    --num_steps 60000 \
    --run_distribute 1 \
    --save_dir "./dgc_832_8p" \
    --ohem_thres 0.7 \
    --ohem_keep 100000 \
    >train_8p.log 2>&1 &
