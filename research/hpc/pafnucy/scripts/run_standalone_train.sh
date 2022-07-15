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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_standalone_train.sh MINDRECORD_PATH"
echo "for example: sh run_standalone_train.sh /data/mindrecord_path device_id"
echo "It is better to use absolute path."
echo "Please pay attention that the dataset should corresponds to dataset_name"
echo "=============================================================================================================="
if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_standalone_train.sh [MINDRECORD_PATH] [DEVICE_ID]"
exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
DEVICEID=$2

export DEVICE_NUM=1
export DEVICE_ID=$DEVICEID
export RANK_ID=0
export RANK_SIZE=1
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
echo "config file path $CONFIG_FILE"

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp -r ../src ./train
cp -r ../scripts/*.sh ./train
cd ./train || exit
mkdir ckpt
echo "start training for device $DEVICE_ID"
env > env.log
python3 -u train.py --config_path=$CONFIG_FILE --mindrecord_path=${DATASET} \
        --enable_modelarts=False  --distribute=False --device_id=$DEVICEID > log.txt 2>&1 &
cd ..