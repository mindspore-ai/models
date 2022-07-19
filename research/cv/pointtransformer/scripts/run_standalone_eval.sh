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

if [ $# -ne 3 ]; then
    echo "Usage:
        bash run_train.sh [DATASET_PATH] [CONFIG] [CHECKPOINT_PATH]"
    exit 1
fi

if [ ! -d $1 ]
    then
        echo "error: DATASET_PATH=$1 is not a directory or file"
    exit 1
fi

if [ ! -f $3 ]
    then
        echo "error: CHECKPOINT_PATH=$3 is not a file"
    exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
BASEPATH=$BASEPATH/..
real_data_path=$(readlink -f $1)

export PYTHONPATH=${BASEPATH}:${PYTHONPATH}
export GLOG_v=3

nohup python -u $BASEPATH/eval.py --dataset_path=$real_data_path --config_path=$2 --pretrain_ckpt=$3 > eval.log 2>&1 &
