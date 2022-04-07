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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH] [CONFIG_FILE] "
exit 1
fi

ulimit -u unlimited
export DATA_PATH=$1
export DEVICE_ID=$2
export CKPT_PATH=$3
export CONFIG_FILE=$4
export DEVICE_NUM=1

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

start=`expr 0 \* $avg`
end=`expr $start \+ $gap`
cmdopt=$start"-"$end

echo "start training"
taskset -c $cmdopt python ../train.py --data_path=$DATA_PATH --device_id=$DEVICE_ID --ckpt_path=./$CKPT_PATH \
--config_path=$CONFIG_FILE > log.txt 2>&1 &