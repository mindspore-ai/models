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

if [ $# != 4 ]
then
    echo "Usage: bash run_eval.sh [CONFIG_FILE] [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo realpath -m "$PWD"/"$1"
  fi
}

CONFIG_PATH=$(get_real_path "$1")
DATASET=$2
CHECKPOINT_PATH=$(get_real_path "$3")
echo "$DATASET"
echo "$CHECKPOINT_PATH"

if [ ! -f "$CHECKPOINT_PATH" ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

if [[ ! -f $CONFIG_PATH ]]
then
    echo "error: CONFIG_FILE=$CONFIG_PATH is not a file"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$4
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

BASE_PATH=$(cd "$(dirname "$0")" || exit; pwd)
cd "$BASE_PATH"/../ || exit

if [ -d "eval$DEVICE_ID" ];
then
    rm -rf ./eval"$DEVICE_ID"
fi

mkdir ./eval"$DEVICE_ID"
cp ./*.py ./eval"$DEVICE_ID"
cp -r ./src ./eval"$DEVICE_ID"
cp -r ./config/*.yaml ./eval"$DEVICE_ID"
cd ./eval"$DEVICE_ID" || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval.py \
    --config_path="$CONFIG_PATH" \
    --dataset="$DATASET" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --device_id="$DEVICE_ID" > log.txt 2>&1 &
cd ..
