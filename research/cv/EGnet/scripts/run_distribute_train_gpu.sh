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

# The number of parameters transferred is not equal to the required number, print prompt information
if [ $# != 2 ]
then 
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_distributed_train_gpu.sh [DEVICE_NUM] [USED_DEVICES]"
    echo "for example: bash run_distributed_train_gpu.sh 8 0,1,2,3,4,5,6,7"
    echo "================================================================================================================="
exit 1
fi

# Get absolute path
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

export RANK_SIZE=$1

export CUDA_VISIBLE_DEVICES=$2

cd $BASE_PATH/..

mpirun -n $RANK_SIZE --allow-run-as-root \
python -u train.py --device_target=GPU  --is_distributed True &> distribute_train.log &

echo "The train log is at ../distribute_train.log."
