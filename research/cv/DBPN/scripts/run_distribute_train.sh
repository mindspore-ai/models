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

if [ $# != 7 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd DBPN"
  echo "Usage: sh run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [TRAIN_GT_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [MODE]"
  echo "bash scripts/run_distribute_train.sh 8 1 ./hccl_8p.json /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR False"
  echo "MODE control the way of trian gan network or only train generator"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export RANK_TABLE_FILE=$3
export RANK_START_ID=4

for((i=0;i<$1;i++))
do
        export DEVICE_ID=$((i + RANK_START_ID))
        rm -rf ./train_parallel$i
        mkdir ./train_parallel$i

        cp -r ../src ./train_parallel$i
        cp -r ../*.py ./train_parallel$i
        cd ./train_parallel$i || exit
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log
        if [ $# == 7 ]
        then
          if [ $7 == "False" ]
          then
              mkdir -p ./train_parallel$i/ckpt/gen
              python ./train_dbpn.py --run_distribute=$2 --device_num=$1 --device_id=$DEVICE_ID \
                                   --train_GT_path=$4 --val_GT_path=$5 --val_LR_path=$6  > paralletrain.log 2>&1 &
          else
              mkdir -p ./train_parallel$i/ckpt/gan
              cp /home/HEU_535/zhijing/DBPN_final/ckpt/gan/dbpn_100.ckpt ./train_parallel$i/ckpt/gan/
              python ./train_dbpngan.py --run_distribute=$2 --device_num=$1 --device_id=$DEVICE_ID \
                                    --train_GT_path=$4 --val_GT_path=$5 --val_LR_path=$6  > paralletrain.log 2>&1 &
          fi
        fi
        cd ..
done
