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

BASE_DIR=$(cd "$(dirname "$0")" || exit; pwd)
RANK_SIZE=$1
RANK_TABLE_FILE=$2

echo $RANK_TABLE_FILE

if [ ! -f ${RANK_TABLE_FILE} ]; then
echo "${RANK_TABLE_FILE} file not exists"
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

  ln -s $BASE_DIR/../data ./device$DEVICE_ID
  cp -r $BASE_DIR/../src  ./device$DEVICE_ID
  cp  $BASE_DIR/../*.py ./device$DEVICE_ID
  cp  $BASE_DIR/../*.ckpt ./device$DEVICE_ID
  cd ./device$DEVICE_ID
  python -u ./train.py  --train_mode 'distribute' config_path $3 > train.log 2>&1 &
  cd ../
done
