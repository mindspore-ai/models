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

if [ $# -lt 6 ]; then
    echo "Usage:
        bash run_distribute_train.sh [DATASET_PATH] [EXP_PATH] [CONFIG] [DEVICE_NUM] [RANK_TABLE_FILE] [PY_ARGS](optional)"
    exit 1
fi

if [ ! -d $1 ] && [ ! -f $1 ]
    then
        echo "error: DATASET_PATH=$1 is not a directory or file"
    exit 1
fi
real_data_path=$(readlink -f $1)

if [ $4 -lt 1 ] || [ $4 -gt 8 ]
    then
        echo "error: DEVICE_NUM=$4 is not in (1-8)"
    exit 1
fi

PY_ARGS=${*:6}

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
BASEPATH=$BASEPATH/..

export PYTHONPATH=${BASEPATH}:${PYTHONPATH}
export RANK_TABLE_FILE=$5
export RANK_SIZE=$4
export GLOG_v=3
export GRAPH_OP_RUN=1
export HCCL_WHITELIST_DISABLE=1

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

echo "Start training, RUN_DISTRIBUTE is True, Save log to $2/train.log"
nohup mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root    \
        python -u $BASEPATH/train.py --run_distribute True                                              \
                                     --dataset_path $real_data_path                                     \
                                     --config_path $3 ${PY_ARGS}                                        \
                                     > train.log 2>&1 &

cd ../
