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

if [ $# != 9 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd DBPN"
  echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [MODEL_TYPE] [TRAIN_GT_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [BATCHSIZE] [MODE]"
  echo "bash run_distribute_train.sh 8 1 ./hccl_8p.json DDBPN /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR 4 False"
  echo "bash run_distribute_train.sh 8 1 ./hccl_8p.json DBPN /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR 4 True"
  echo "MODE control the way of trian gan network or only train generator"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export RANK_TABLE_FILE=$3
export RANK_START_ID=0
export RANK_SIZE=$1

for((i=0;i<$1;i++))
do
        export DEVICE_ID=$((i + RANK_START_ID))
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log
        if [ $9 == "False" ]
        then
            rm -rf ./train_dbpn_parallel$i
            mkdir ./train_dbpn_parallel$i
            cp -r ../src ./train_dbpn_parallel$i
            cp -r ../*.py ./train_dbpn_parallel$i
            cd ./train_dbpn_parallel$i || exit
            python train_dbpn.py --device_num=$1 --run_distribute=$2 --model_type=$4 --device_id=$DEVICE_ID \
                                 --train_GT_path=$5 --val_GT_path=$6 --val_LR_path=$7 --batchSize=$8 > paralletrain.log 2>&1 &
        else
            rm -rf ./train_dbpngan_parallel$i
            mkdir ./train_dbpngan_parallel$i
            cp -r ../src ./train_dbpngan_parallel$i
            cp -r ../*.py ./train_dbpngan_parallel$i
            cd ./train_dbpngan_parallel$i || exit
            python train_dbpngan.py --device_num=$1 --run_distribute=$2 --model_type=$4 --device_id=$DEVICE_ID \
                                  --train_GT_path=$5 --val_GT_path=$6 --val_LR_path=$7 --batchSize=$8 > paralletrain.log 2>&1 &
        fi
        cd ..
done
