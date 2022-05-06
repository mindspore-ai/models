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
    echo "Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES] [DATA_URL] [TRAIN_URL]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$3
PATH2=$(get_real_path $4)

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
    exit 1
fi

if [ ! -d $PATH1 ]
then
    echo "error: TRAIN_URL=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: DATA_URL=$PATH2 is not a directory"
exit 1
fi

echo "DEVICE_NUM:" $1
echo "CUDA_VISIBLE_DEVICES:" $2
echo "DATA_URL:" $PATH1
echo "TRAIN_URL:" $PATH2

ulimit -c unlimited
export DEVICE_NUM=$1
export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES=$2

rm -rf ./train_parallel
mkdir ./train_parallel
mkdir ./train_parallel/scripts
cp ../*.py ./train_parallel
cp *.sh ./train_parallel/scripts
cp -r ../src ./train_parallel
cp -r ../gpu_infer ./train_parallel
cd ./train_parallel || exit
mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root python train.py --device_target GPU --run_distribute True\
                      --data_url $PATH1 --train_url $PATH2 > output.train.dis_log 2>&1 &
cd ..




