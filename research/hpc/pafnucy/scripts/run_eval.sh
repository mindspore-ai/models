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
echo "sh run_eval.sh MINDRECORD_PATH CKPT_PATH DEVICE_ID"
echo "for example: bash run_eval.sh /data/path checkpoint/path device_id"
echo "It is better to use absolute path."
echo "Please pay attention that the dataset should corresponds to dataset_name"
echo "=============================================================================================================="
if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_eval.sh [MINDRECORD_PATH] [CKPT_PATH] [DEVICE_ID]"
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
CKPT=$(get_real_path $2)
DEVICEID=$3

export DEVICE_NUM=1
export DEVICE_ID=$DEVICEID
export RANK_ID=0
export RANK_SIZE=1
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
echo "config file path $CONFIG_FILE"

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cp -r ../scripts/*.sh ./eval
cd ./eval || exit
mkdir ckpt
echo "start training for device $DEVICE_ID"
env > env.log
python3 -u eval.py --config_path=$CONFIG_FILE --mindrecord_path=${DATASET} \
  --enable_modelarts=False  --ckpt_file=$CKPT --device_id=$DEVICEID > log.txt 2>&1 &
cd ..