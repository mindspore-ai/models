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

if [ $# != 5 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash ./scripts/run_standalone_train.sh [PLATFORM] [DEVICE_ID] [CONTENT_PATH] [STYLE_PATH] [CKPT_PATH]"
    echo "for example: bash ./scripts/run_train.sh 'Ascend' 1 /home/ArbitraryStyleTransfer/dataset/content /home/ArbitraryStyleTransfer/dataset/style /home/ArbitraryStyleTransfer/pretrained_model/
"
    echo "=============================================================================================================="
exit 1
fi

export RANK_ID=$2
export PLATFORM=$1
export DEVICE_ID=$2
export CONTENT_PATH=$3
export STYLE_PATH=$4
export CKPT_PATH=$5


python train.py \
  --run_offline 1\
  --run_distribute 0\
  --device_num 1\
  --platform $PLATFORM \
  --device_id $DEVICE_ID \
  --content_path $CONTENT_PATH \
  --style_path $STYLE_PATH \
  --ckpt_path $CKPT_PATH > log 2>&1 &

