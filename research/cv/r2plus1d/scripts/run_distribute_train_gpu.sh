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

DIR="$( cd "$( dirname "$0"  )" && pwd  )"

# help message
if [ $# != 2 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh [num_devices] [cuda_visible_devices(0,1,2,3,4,5,6,7)]"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530

export CUDA_VISIBLE_DEVICES=$2

cd $DIR/../ || exit
mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
  python train.py --device_target=GPU --is_distributed=1 &> distribute_train_gpu_log.txt &
