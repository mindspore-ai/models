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

if [ $# != 3 ]
then
    echo "Usage: bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

BASE_DIR=$(cd "$(dirname "$0")" || exit; pwd)
RANK_SIZE=$1
RANK_TABLE_FILE=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)

if [ ! -f ${RANK_TABLE_FILE} ]; then
echo "rank table ${RANK_TABLE_FILE} file not exists"
exit
fi

if [ ! -f ${CONFIG_PATH} ]; then
echo "config path ${CONFIG_PATH} file not exists"
exit
fi

export RANK_TABLE_FILE=${RANK_TABLE_FILE}
export RANK_SIZE=${RANK_SIZE}
rank_start=0

for((i=0;i<${RANK_SIZE};i++))
do
  export DEVICE_ID=$((rank_start + i))
  export RANK_ID=$i
  echo "rank $RANK_ID device $DEVICE_ID"
  rm -rf device$DEVICE_ID
  mkdir device$DEVICE_ID

  cp -r $BASE_DIR/../src  ./device$DEVICE_ID
  cp  $BASE_DIR/../*.py ./device$DEVICE_ID
  cp  $BASE_DIR/../*.yaml ./device$DEVICE_ID
  cd ./device$DEVICE_ID
  python -u ./train.py \
      --train_mode="distribute" \
      --device_target="Ascend" \
      --config_path=$CONFIG_PATH > train.log 2>&1 &
  cd ../
done
