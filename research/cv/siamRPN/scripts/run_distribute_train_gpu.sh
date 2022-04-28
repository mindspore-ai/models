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

if [ $# != 2 ]
then 
    echo "Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]"
exit 1
fi


DEVICE_NUM=$1
echo $DEVICE_NUM

export DEVICE_NUM=$1
export RANK_SIZE=$DEVICE_NUM
export CUDA_VISIBLE_DEVICES="$2"


nohup mpirun -n $DEVICE_NUM --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python -u train.py  --device_target="GPU" --is_parallel=True > train_gpu.log 2>&1 &
