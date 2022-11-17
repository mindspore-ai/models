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
    echo "Usage: bash run_eval.sh [RUN_OFFLINE] [VAL_PATH] [VAL_GT_PATH] [CKPT_PATH]"
exit 1
fi

ulimit -u unlimited
export DEVICE_ID=0
export RANK_SIZE=1
export RUN_OFFLINE=$1
export VAL_PATH=$2
export VAL_GT_PATH=$3
export CKPT_PATH=$4

if [ -d "eval" ];
then
    rm -rf ./eval
fi

mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"
python eval.py --run_offline=$RUN_OFFLINE --val_path=$VAL_PATH \
               --val_gt_path=$VAL_GT_PATH --ckpt_path=$CKPT_PATH &> log &
cd ..
