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
echo "bash run_distribute_train.sh MINDRECORD_PATH RANK_TABLE DEVICE_NUM"
echo "For example: bash run_distribute_train.sh /path/mindrecord_path /path/rank_table device_num"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_distribute_train.sh [MINDRECORD_PATH] [RANK_TABLE] [DEVICE_NUM]"
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

ulimit -u unlimited
export DEVICE_NUM=$3
export RANK_SIZE=$3
export RANK_TABLE_FILE=$(get_real_path $2)
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
echo "config file path $CONFIG_FILE"

echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir ./device$i
    cp ../*.py ./device$i
    cp ../*.yaml ./device$i
    cp -r ../src ./device$i
    cp -r ../scripts/*.sh ./device$i
    cd ./device$i || exit
    mkdir ckpt
    echo "start training for device $DEVICE_ID"
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    nohup python3 -u train.py --config_path=$CONFIG_FILE --mindrecord_path=${DATASET} \
                              --enable_modelarts=False  --batch_size=8 \
                              --distribute=True > log.txt 2>&1 &
    echo "$i finish"
    cd ../
done
