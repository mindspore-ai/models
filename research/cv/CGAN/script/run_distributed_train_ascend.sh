#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
    echo "Usage: sh run_distributed_train_ascend.sh [DATA_PATH] [OUTPUT_PATH] [RANK_TABLE] [DEVICE_NUM]"
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
export RANK_TABLE_FILE=$3
export DEVICE_NUM=$4
export RANK_SIZE=$DEVICE_NUM
export HCCL_CONNECT_TIMEOUT=600
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

# remove old train_parallel files
rm -rf ./train_parallel
mkdir ./train_parallel
echo "device count=$DEVICE_NUM"

i=0
while [ $i -lt ${DEVICE_NUM} ]; do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))

    # mkdirs
    mkdir ./train_parallel/$i
    mkdir ./train_parallel/$i/src

    # move files
    cp ../*.py ./train_parallel/$i
    cp ../src/*.py ./train_parallel/$i/src

    # goto the training dirs of each training
    cd ./train_parallel/$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    # input logs to env.log
    env > env.log
    python -u train.py --device_target=Ascend --device_id=$i --distribute=True --data_path=$DATASET --output_path=$OUTPUT_PATH > log 2>&1 &
    cd ../..
    i=$((i + 1))
done
