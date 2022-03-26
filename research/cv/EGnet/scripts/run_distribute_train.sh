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


set -e
RANK_SIZE=$1
RANK_TABLE_FILE=$2

if [ ! -f ${RANK_TABLE_FILE} ]; then
echo "file not exists"
exit
fi
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
export RANK_SIZE=${RANK_SIZE}

for((i=0;i<${RANK_SIZE};i++))
do
  rm -rf device$i
  mkdir device$i
  cp -r ../src ./device$i
  cp -r ../model_utils ./device$i
  cp -r ../data ./device$i
  cp ../sal2edge.py ./device$i
  cp ../train.py ./device$i
  cp ../default_config.yaml ./device$i
  cd ./device$i
  export DEVICE_ID=$i
  export RANK_ID=$i
  python -u ./train.py  --is_distributed True    > train.log 2>&1 &
  cd ../
done
