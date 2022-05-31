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
if [ $# != 4 ]
then
    echo "Usage: bash run_distributed_train_gpu.sh [DATA_PATH] [OUTPUT_PATH] [CUDA_VISIBLE_DEVICES] [DEVICE_NUM]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)
OUTPUT_PATH=$(get_real_path $2)

export DATASET=$DATA_PATH
export OUTPUT_PATH=$OUTPUT_PATH
export CUDA_VISIBLE_DEVICES=$3
export DEVICE_NUM=$4
export RANK_SIZE=$4

# remove old train_parallel files
rm -rf ./train_parallel
# mkdirs
mkdir ./train_parallel
mkdir ./train_parallel/src

# move files
cp ../*.py ./train_parallel/
cp ../src/*.py ./train_parallel/src

# goto the training dirs of each training
cd ./train_parallel/ || exit

echo "start training on $DEVICE_NUM devices"
echo "the results will saved in $OUTPUT_PATH"

# input logs to env.log
mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root\
 python train.py --data_path=$DATASET --output_path=$OUTPUT_PATH --device_target=GPU --distribute=True > log 2>&1 &


