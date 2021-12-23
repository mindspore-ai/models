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

if [ $# != 2 ]; then
  echo "Usage: bash run_eval_ascend.sh [DATA_DIR] [CKPT]"
  exit 1
fi

DATAPATH=$1
echo "info: load data from $DATAPATH"
CKPT=$2

if [ ! -d $DATAPATH ]; then
  echo "error: DATA_DIR=$DATAPATH is not a directory"
  exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

echo "start evaluation for device $DEVICE_ID"
env > env.log

python -u eval.py \
    --device_target "Ascend" \
    --dir_data $DATAPATH \
    --batch_size 1 \
    --test_only True\
    --ext "img" \
    --data_test B100 \
    --ckpt_path $CKPT \
    --task_id 0 \
    --scale 4
cd ..