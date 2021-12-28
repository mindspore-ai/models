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
if [ $# != 2 ] && [ $# != 3 ]; then
    echo "Usage: bash scripts/run_train_ascend.sh [INPUT_DIR] [INPUT_NAME] [DEVICE_ID]"
    echo "DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
fi
input_dir=$1
input_name=$2
device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi
nohup python3 -u train.py --input_dir=$input_dir --input_name=$input_name --device_id=$device_id > log 2>&1 &