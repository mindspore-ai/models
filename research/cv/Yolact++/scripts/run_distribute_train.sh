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

if [ $# -lt 2 ]
then
  echo "####################################################"
  echo "Usage: bash run_distribute_train.sh RANK_TABLE_FILE DEVICE_NUMS"
  echo "Example:"
  echo "bash run_distribute_train.sh hccl_8p.json 8"
  echo "use hccl_8p.json as RANK_TABLE_FILE and 8 devices"
  echo "####################################################"
  exit 1
fi

get_absolute_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_absolute_path $1)
DEVICE_NUM=$2
echo "you want to run on $DEVICE_NUM devices"

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 


export SERVER_ID=0
ulimit -u unlimited
rank_start=$((DEVICE_NUM * SERVER_ID))
first_device=0

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$((first_device+i))
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    echo "$i"
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../weights ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit

    echo "[+] start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log

    nohup python train.py \
    --device_num=$DEVICE_NUM \
    --rank_id=$RANK_ID \
    --device_id=$DEVICE_ID > log 2>&1 &

    cd ..
done
