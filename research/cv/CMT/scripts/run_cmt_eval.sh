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
if [ $# -lt 3 ]
then
    echo "Usage: bash ./scripts/run_cmt_eval.sh [DATA_PATH] [PLATFORM] [CHECKPOINT_PATH]"
exit 1
fi

DATA_PATH=$1
PLATFORM=$2
CHECKPOINT_PATH=$3

rm -rf evaluation
mkdir ./evaluation
cd ./evaluation || exit
echo  "start training for device id $DEVICE_ID"
env > env.log
python eval.py --model cmt --dataset_path=$DATA_PATH --platform=$PLATFORM --checkpoint_path=$CHECKPOINT_PATH > eval.log 2>&1 &
cd ../