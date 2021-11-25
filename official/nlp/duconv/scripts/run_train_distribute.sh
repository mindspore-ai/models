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

"""
train distribute model 

"""
if [ $# != 4 ]; then
  echo "Usage: bash run_distribute_train.sh [TASK] [DATA_PATH] [RANK_TABLE_FILE]"
  exit 1
fi

if [ ! -f $3 ]
then
    echo "error: RANK_TABLE_FILE=$3 is not a file"
exit 1
fi

if [ ! -f $2 ]
then
    echo "error: RANK_TABLE_FILE=$2 is not a file"
exit 1
fi


TASK_NAME=$1
DATA_PATH=$2
export RANK_TABLE_FILE=$3
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=600
DEVICE_NUM=$RANK_SIZE
echo "DEVICE_NUM is $DEVICE_NUM"
OUTPUT_PATH=$4
cd ..
PWD_IDR=`pwd`
rm -rf $OUTPUT_PATH
mkdir $OUTPUT_PATH
save_path=$PWD_IDR/$OUTPUT_PATH
for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    cd $save_path
    rm -rf LOG$i
    mkdir LOG$i
    cd ./LOG$i || exit
    echo "start training for device $DEVICE_ID"
    cd ../../
    python train.py --epoch=30 \
                    --task_name=${TASK_NAME} \
                    --max_seq_length=256 \
                    --batch_size=128 \
                    --train_data_file_path=$DATA_PATH \
                    --run_distribute=True\
                    --device_num=$DEVICE_NUM\
                    --device_id=$DEVICE_ID\
                    --save_checkpoint_path=$save_path/checkpoint\
                    --config_path=default_config.yaml > $save_path/LOG$i/log.txt 2>&1 &
    
done
