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
    echo "Usage: sh run_distribute_train_for_gpu.sh [DATASET_PATH]"
exit 1
fi

# check dataset path
if [ ! -d $1 ]
then
    echo "error: DATASET_PATH=$1 is not a directory"    
exit 1
fi

DATASET_PATH=$1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
  python ../train.py --is_distributed=True --device_target='GPU' --num_epoch_per_decay 2.2 --dataset_path $DATASET_PATH > train.log 2>&1 &
