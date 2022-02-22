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


if [[ $# -ne 2 ]]; then
    echo "Usage: bash ./scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DEVICE_START]"
exit 1
fi

device_start=$2
rank_size=$1
device_end=$(($device_start+$rank_size-1))
gpus=""
for i in $(eval echo {$device_start..$device_end})
    do
        gpus="$gpus,$i"
done
gpus="${gpus:1}"
export CUDA_VISIBLE_DEVICES=$gpus
rm -rf logs
mkdir ./logs
nohup mpirun -n $rank_size --allow-run-as-root python train.py --run_distribute True > ./logs/train.log 2>&1 &
echo $! > ./logs/train.pid

