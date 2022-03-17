#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
if [ $# != 3 ]
then
    echo "Usage: bash run_eval.sh [DEVICE_TARGET] [DATASET_PATH] [CHECKPOINT_PATH]"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

if [ ! -f $3 ]
then
    echo "error: CHECKPOINT_PATH=$3 is not a file"
exit 1
fi

export RANK_SIZE=1

python $BASE_PATH/../eval.py --target=$1 --dataset_path=$2 --resume=$3 &> eval.log &
