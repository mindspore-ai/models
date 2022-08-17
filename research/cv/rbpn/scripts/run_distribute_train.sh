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


if [ $# != 6 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd RBPN"
  echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [DATA_DIR] [FILE_LIST] [BATCHSIZE] "
  echo "bash run_distribute_train.sh 8 1 ./hccl_8p.json /data/RBPN_data/vimeo_septuplet/sequences /data/RBPN_data/vimeo_septuplet/sep_trainlist.txt  4 "
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export RANK_TABLE_FILE=$3
export RANK_START_ID=0
export RANK_SIZE=$1
echo "lets begin!!!!XD"

for((i=0;i<$1;i++))
do
        export DEVICE_ID=$((i + RANK_START_ID))
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log

        rm -rf ./train_rbpn_parallel$i
        mkdir ./train_rbpn_parallel$i
        cp -r ../src  ./train_rbpn_parallel$i
        cp -r ../*.py ./train_rbpn_parallel$i
        cp -r ../*.so ./train_rbpn_parallel$i
        cd ./train_rbpn_parallel$i || exit
        python train.py --device_num=$1 --run_distribute=$2  --data_dir=$4   --file_list=$5  --batchSize=$6  --device_id=$DEVICE_ID  > paralletrain.log 2>&1 &
        cd ..
done
