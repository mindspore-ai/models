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
    echo "Usage: bash run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT] [DEVICE_TARGET]"
exit 1
fi

# check device id
if [ "$1" -eq "$1" ] 2>/dev/null
then
    if [ $1 -lt 0 ] || [ $1 -gt 7 ]
    then
        echo "error: DEVICE_ID=$1 is not in (0-7)"
    exit 1
    fi
else
    echo "error: DEVICE_ID=$1 is not a number"
exit 1
fi

# check dataset path
if [ ! -d $2 ]
then
    echo "error: DATASET_PATH=$2 is not a directory"
exit 1
fi

# check checkpoint file
if [ ! -f $3 ]
then
    echo "error: CHECKPOINT=$3 is not a file"
exit 1
fi

# check device target
if [ "$4" != "Ascend" ] && [ "$4" != "GPU" ]
then
  echo "error: DEVICE_TARGET is not in [Ascend, GPU]"
  exit 1
fi

python ../eval.py --device_id=$1 --dataset_path=$2 --checkpoint=$3 --enable_checkpoint_dir=False --device_target=$4 > ./eval.log 2>&1 &
