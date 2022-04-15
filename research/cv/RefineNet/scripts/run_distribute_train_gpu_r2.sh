#! /bin/bash
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
if [ $# -ne 3 ]
then
    echo "Usage: bash scripts/run_distribute_train_gpu_r1.sh [DATASET_PATH] [PRETRAINED_PATH] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$1
if [ ! -f $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a file"
exit 1
fi

PRETRAINED_PATH=$(get_real_path $2)
echo $PRETRAINED_PATH
if [ ! -f $PRETRAINED_PATH ]
then
    echo "error: PRETRAINED_PATH=$PRETRAINED_PATH is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export CUDA_VISIBLE_DEVICES="$3"

rm -rf ./train_parallel_r2
mkdir ./train_parallel_r2
cp ../*.py ./train_parallel_r2
cp *.sh ./train_parallel_r2
cp -r ../src ./train_parallel_r2
cd ./train_parallel_r2 || exit
echo "start training "
env > env.log

if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n 8 \
    python train.py  \
    --is_distribute  \
    --data_file=$DATASET_PATH  \
    --ckpt_pre_trained=$PRETRAINED_PATH  \
    --base_lr=0.00004  \
    --batch_size=8  \
    --device_target='GPU' &> log &
fi
