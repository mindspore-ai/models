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

# Get absolute path
echo "Usage: bash run_distribute_gpu.sh"

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

cd $BASE_PATH/..

mpirun --allow-run-as-root -n 8 python train.py --train_mode 'distribute'  &> distribute.log 2>&1 &

echo "The train log is at ../distribute.log."
