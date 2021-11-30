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

# help message
# "Usage: bash run_distribute_train_gpu.sh"

DIR="$( cd "$( dirname "$0"  )" && pwd  )"

ulimit -c unlimited
ulimit -n 65530

rm -rf $DIR/../gpu_work_space
mkdir $DIR/../gpu_work_space
cp -r $DIR/../src $DIR/../gpu_work_space
cp $DIR/../train.py $DIR/../default_config.yaml $DIR/../gpu_work_space
cd $DIR/../gpu_work_space
echo "start training"
mpirun --allow-run-as-root -n 8 python ./train.py \
       --device_target=GPU \
       --is_distributed=1 > log.txt 2>&1 &
