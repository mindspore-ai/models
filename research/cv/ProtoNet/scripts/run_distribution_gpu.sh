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
if [ $# != 4 ]
then
    echo "Usage: sh run_distribution_gpu.sh [DATA_PATH] [TRAIN_CLASS] [EPOCHS] [DEVICE_NUM]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
export DATA_PATH=$(get_real_path $1) # dataset path
export TRAIN_CLASS=$2 # train class, propose 20
export EPOCHS=$3 # num of epochs
export RANK_SIZE=$4 # device_num

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a directory"
exit 1
fi

rm -rf distribute_output
mkdir distribute_output

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python ../train.py --dataset_root=$DATA_PATH \
                --device_target="GPU" \
                --classes_per_it_tr=$TRAIN_CLASS \
                --experiment_root="./distribute_output" \
                --epochs=$EPOCHS > distribute_log 2>&1 &
