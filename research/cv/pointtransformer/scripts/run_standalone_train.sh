#!/usr/bin/env bash
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

if [ $# -lt 4 ]; then
    echo "Usage:
        bash run_standalone_train.sh [DATASET_PATH] [EXP_PATH] [DEVICE_ID] [CONFIG] [PY_ARGS](optional)"
    exit 1
fi

if [ ! -d $1 ] && [ ! -f $1 ]
    then
        echo "error: DATASET_PATH=$1 is not a directory or file"
    exit 1
fi

if [ $3 -lt 0 ] || [ $3 -gt 7 ]
    then
        echo "error: DEVICE_ID=$3 is not in (0-7)"
    exit 1
fi

PY_ARGS=${*:5}

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
BASEPATH=$BASEPATH/..
real_data_path=$(readlink -f $1)

export PYTHONPATH=${BASEPATH}:${PYTHONPATH}
export GLOG_v=3
export DEVICE_ID=$3
export RANK_ID=0

if [ -d $2 ];
then
    echo "Remove previous log directory $2:"
    rm -rf $2
fi
mkdir $2
cd $2 || exit

ulimit -u unlimited
ulimit -n 65536
env > env.log

echo "Start standalon training, Save log to $2/train.log"
nohup python -u $BASEPATH/train.py          \
            --run_distribute False          \
            --dataset_path $real_data_path  \
            --config_path $4 ${PY_ARGS}     \
            > train.log 2>&1 &
cd ../
