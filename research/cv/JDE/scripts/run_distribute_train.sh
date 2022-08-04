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
# This file was copied from project [ascend][modelzoo-his]
if [[ $# -ne 5 ]]; then
    echo "Usage: bash ./scripts/run_distribute_train_npu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT] [RANK_TABLE_FILE]"
exit 1;
fi

export RANK_SIZE=$1
export RANK_TABLE_FILE=$5
export GLOG_v=3     # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR
if [ !  -d "$2" ]; then
  mkdir "$2"
fi

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

for((i=0; i<$1; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=$i
    export RANK_ID=$i

    rm -rf ./"$2"/train_parallel$i
    mkdir ./"$2"/train_parallel$i
    cp ./*.py ./"$2"/train_parallel$i
    cp ./*.yaml ./"$2"/train_parallel$i
    cp -r ./src ./"$2"/train_parallel$i
    cp -r ./cfg ./"$2"/train_parallel$i
    cp -r ./data ./"$2"/train_parallel$i
    cp $3 ./"$2"/train_parallel$i
    cp $4 ./"$2"/train_parallel$i
    cd ./"$2"/train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    taskset -c $cmdopt python train.py \
        --device_target="Ascend" \
        --device_id=$i \
        --logs_dir=$2 \
        --ckpt_url=$3 \
        --dataset_root=$4 >devide_$i.log 2>&1 &
    cd ..
    cd ..
done