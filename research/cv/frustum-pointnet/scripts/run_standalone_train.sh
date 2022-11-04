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

# ulimit -m unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ $# != 3 ]
then
    echo "Usage: bash scripts/run_standalone_train.sh [LOG_DIR] [DEVICE_TARGET] [DEVICE_ID]"
    echo "============================================================"
    echo "[LOG_DIR]: The path to save the train and evaluation log."
    echo "[DEVICE_TARGET]: Platform used."
    echo "[DEVICE_ID]: card id want to use."
    echo "============================================================"
exit 1
fi

if [ $# -ge 1 ]
then
    LOG_DIR=$1
    mkdir $LOG_DIR
    if [ ! -d $LOG_DIR ]
    then
        echo "error: DATA_PATH=$LOG_DIR is not a directory"
    exit 1
    fi
fi
export DEVICE_ID=$3
echo "start training, log will output to $LOG_DIR/train.log and the model file will save to $LOG_DIR/Fpointnet*.ckpt by default"
python -u train_net.py --log_dir=$LOG_DIR --device_target=$2 > $LOG_DIR/train.log 2>&1 &
echo 'running'
