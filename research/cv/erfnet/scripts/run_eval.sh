#! /bin/bash
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
  echo "Usage: bash scripts/run_eval.sh DEVICE_TARTGET DATASET_PATH CHECKPOINT_PATH DEVICE_ID"
  echo "Example: bash scripts/run_eval.sh [Ascend/GPU] /home/name/cityscapes distribute_train/checkpoint 0"
  exit 1
fi

echo "DEVICE_TARTGET: $1"
echo "CITYSCAPES DATASET PATH: $2"
echo "CHECKPOINT_PATH: $3"
echo "DEVICE_ID: $4"

DEVICE_TARTGET=$1
DATASET_PATH=$2
export RANK_SIZE=1
CHECKPOINT_PATH=$3
DEVICE_ID=$4

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

python -u $BASE_PATH/../eval.py \
    --data_path=$DATASET_PATH \
    --run_distribute=false \
    --device_target=$DEVICE_TARTGET \
    --encode=false \
    --model_root_path=$CHECKPOINT_PATH \
    --device_id=$DEVICE_ID \
    > log_eval.txt 2>&1 &

