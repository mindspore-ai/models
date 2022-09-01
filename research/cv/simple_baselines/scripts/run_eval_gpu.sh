#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-1.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# -ne 1 ]; then
    echo "Please run the script as: "
    echo "bash scripts/run_eval_gpu.sh [DEVICE_ID]"
    echo "For example: bash scripts/run_eval_gpu.sh 0"
    echo "It is better to use the absolute path."
    echo "========================================================================"
    exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$1

rm -rf ./eval
mkdir ./eval
cp ./*.py ./eval
cp -r ./src ./eval
cd ./eval || exit

echo "start evaluation on GPU device $DEVICE_ID"
env > env.log

python eval.py --device_target="GPU" --device_id=$DEVICE_ID > eval.log 2>&1 &
