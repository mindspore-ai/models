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

if [ $# != 3 ] && [ $# != 4 ]
then
  echo "Usage: bash train_distribute_ascend.sh [bignet] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $2)
PATH2=$(get_real_path $3)

if [ $# == 4 ]
then 
    PATH3=$(get_real_path $4)
fi

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi 

if [ $# == 4 ] && [ ! -f $PATH3 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID, device_num $DEVICE_NUM"
    env > env.log
    if [ $# == 3 ]
    then    
        python train.py \
            --distributed \
            --device_num=$DEVICE_NUM \
            --model $1 \
            --data_path $PATH2  \
            --num-classes 1000 \
            --channels 16,24,40,80,112,160 \
            --layers 1,2,2,4,2,5 \
            --batch-size 256 \
            --drop 0.2 \
            --drop-path 0 \
            --opt rmsprop \
            --opt-eps 0.001 \
            --lr 0.048 \
            --decay-epochs 2.4 \
            --warmup-lr 1e-6 \
            --warmup-epochs 3 \
            --decay-rate 0.97 \
            --ema-decay 0.9999 \
            --weight-decay 1e-5 \
            --per_print_times 100 \
            --epochs 300 \
            --ckpt_save_epoch 5 \
            --workers 8 \
            --amp_level O2 > $1.log 2>&1 &
    fi
    
    if [ $# == 4 ]
    then
        python train.py --model=$1 --distributed --device_num=$DEVICE_NUM --data_path=$PATH2 --pre_trained=$PATH3 &> log &
    fi

    cd ..
done
