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
    echo "Usage: bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

RANK_TABLE_PATH=$(get_real_path $1)
echo $RANK_TABLE_PATH

if [ ! -f $RANK_TABLE_PATH ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
exit 1
fi

DATASET_PATH=$2
if [ ! -f $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a file"
exit 1
fi

PRETRAINED_PATH=$(get_real_path $3)
echo $PRETRAINED_PATH
if [ ! -f $PRETRAINED_PATH ]
then
    echo "error: PRETRAINED_PATH=$PRETRAINED_PATH is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_PATH

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --device_id=$i --rank=$i --is_distribute --data_file=$DATASET_PATH --ckpt_pre_trained=$PRETRAINED_PATH --base_lr=0.00032 --batch_size=32 &> log &
    cd ..
done

