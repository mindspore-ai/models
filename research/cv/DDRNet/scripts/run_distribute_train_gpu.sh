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

if [ $# -eq 1 ]
then
    echo "Usage: bash ./scripts/run_distribute_train_gpu.sh [CONFIG_PATH]"
exit 1
fi

CONFIG_PATH=$1
export RANK_SIZE=8
export DEVICE_NUM=8

rm -rf train_parallel
mkdir ./train_parallel
cp -r ./src ./train_parallel
cp  *.py ./train_parallel
cd ./train_parallel || exit
echo "start training, device $DEVICE_NUM"
env > env.log
mpirun --allow-run-as-root -n $DEVICE_NUM \
python -u ../train.py \
    --device_target "GPU" \
    --ddr_config=$CONFIG_PATH > log.txt 2>&1 &
cd ../
