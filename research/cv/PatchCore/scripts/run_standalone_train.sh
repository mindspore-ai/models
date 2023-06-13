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

if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_train.sh DATA_PATH DEVICE_ID PRETRAINED_PATH CATEGORY"
    echo "For example: bash run_train.sh /path/dataset /path/pretrained_path category 0"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}
DATA_PATH=$(get_real_path "$1")
CKPT_APTH=$(get_real_path "$2")
export DATA_PATH=$DATA_PATH

train_path=train_$3
if [ -d $train_path ];
then
    rm -rf ./$train_path
fi
mkdir ./$train_path
cd ./$train_path
env > env0.log
echo "[INFO] start train dataset $4."
python ../../train.py --dataset_path $DATA_PATH --isModelArts False --pre_ckpt_path $CKPT_APTH --category $3  --device_id $4 &> train.log

if [ $? -eq 0 ];then
    echo "[INFO] training success"
else
    echo "[ERROR] training failed"
    exit 2
fi
echo "[INFO] finish"
cd ../
