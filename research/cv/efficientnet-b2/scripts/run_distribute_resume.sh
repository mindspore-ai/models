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

# when use this shell to resume training,you should rename the ckpt to the interrupt epoch
# for example:
# mv best_acc.ckpt Efficientnet_b2-rank0-100_1251.ckpt
if [ $# != 5 ]
then
    echo "usage sh run_distribute_resume.sh [device_num] [device_id(0,1,2,3,4,5,6,7)] [dataset_path] [resume] [device_target] "
exit 1
fi

# check dataset file
if [ ! -d $3 ]
then
    echo "error: DATASET_PATH=$3 is not a directory"
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

export DEVICE_NUM=$1
export RANK_SIZE=$1

cd ../train || exit

export CUDA_VISIBLE_DEVICES="$2"
mpirun --allow-run-as-root -n $1  --output-filename log_output --merge-stderr-to-stdout \
python ${BASEPATH}/../train.py \
    --run_distribute True \
    --dataset_path $3 \
    --resume $4 \
    --device_target $5 > train.log 2>&1 &


