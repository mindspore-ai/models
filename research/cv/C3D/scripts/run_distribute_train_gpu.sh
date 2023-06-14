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
# bash run_distribute_train_gpu.sh

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh [CONFIG PATH]"
echo "For example: bash sh run_distribute_train_gpu.sh  CONFIG_PATH"
echo "=============================================================================================================="
set -e
if [ $# != 1 ]
then
  echo "Usage: bash run_distribute_train_gpu.sh CONFIG_PATH"
exit 1
fi

CONFIG_PATH=$1

export CONFIG_PATH=${CONFIG_PATH}

echo "distribute train begin."
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 \
python train.py --config_path=$CONFIG_PATH --is_distributed=1 > train_distributed.log 2>&1 &
echo "Training background..."