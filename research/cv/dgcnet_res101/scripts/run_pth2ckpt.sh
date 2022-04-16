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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh [LOAD_PATH] [SAVE_PATH]"
echo "For example: bash run.sh /path/pth /path/containing/ckpt "
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

export LOAD_PATH=$1
export SAVE_PATH=$2

echo "start converting"
    python ./src/pth2ckpt.py --load_path $1 \
    --save_path $2 \
    >converted.log 2>&1 &
