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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: bash run_distribution_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [CIFAR10_DATA_PATH] [OUTPUT_PATH]"
exit 1
fi

if [ $1 -lt 1 ] || [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi

export RANK_SIZE=$1
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="$2"

CIFAR10_DATA_PATH=$3
OUTPUT_PATH=$4


mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python ${BASEPATH}/../train.py --device_target=GPU  --data_url $CIFAR10_DATA_PATH \
--train_url $OUTPUT_PATH --optimizer SGD --load_weight None --no_top False \
--learning_rate 0.075 --batch_size 32 --amp_level=O2 > log 2>&1 &