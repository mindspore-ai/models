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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh RANK_TABLE_FILE RANK_SIZE RANK_START /path/dataset"
echo "For example: bash run.sh  /path/rank_table.json 8 0 /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
execute_path=$(pwd)
echo ${execute_path}

export RANK_TABLE_FILE=$1
export RANK_SIZE=$2
DEVICE_START=$3
DATASET_PATH=$4
for((i=0;i<$RANK_SIZE;i++));
do
  export RANK_ID=$i
  export DEVICE_ID=$((DEVICE_START + i))
  echo "Start training for rank $i, device $DEVICE_ID."
  if [ -d ${execute_path}/device${DEVICE_ID} ]; then
      rm -rf ${execute_path}/device${DEVICE_ID}
    fi
  mkdir ${execute_path}/device${DEVICE_ID}
  cp -f train.py ${execute_path}/device${DEVICE_ID}
  cp -rf src ${execute_path}/device${DEVICE_ID}
  cd ${execute_path}/device${DEVICE_ID} || exit
  python3.7 -u train.py --distributed 'True' --dataset_path ${DATASET_PATH} > log$i 2>&1 &
  cd ..
done
