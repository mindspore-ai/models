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

echo "=============================================================================================================="
echo "Please run the script at the diractory same with test.py: "
echo "bash scripts/run_test_ascend.sh data_url checkpoint_path device_id"
echo "For example: bash scripts/run_test_ascend.sh /path/dataset/ /path/resnet50_300-23.ckpt 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

data_url=$1
checkpoint_path=$2
device_id=$3

EXEC_PATH=$(pwd)

python ${EXEC_PATH}/test.py --data_url=$data_url --checkpoint_path=$checkpoint_path --device_id=$device_id > test.log 2>&1 &
