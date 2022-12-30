#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 1 ]
then
    echo "Usage: bash run_export_ascend.sh [DATASET]"
exit 1
fi

DATASET=$1

export GLOG_v=3

echo "DATASET: $DATASET"

if [ ! -e "checkpoints/$DATASET.ckpt" ]
then
    echo "ckpt file not exists"
exit 1
fi

if [ ! -d "mindir" ]
then
    mkdir mindir
fi

if [ ! -d "logs" ]
then
    mkdir logs
fi

echo "Start exporting..."
nohup python -u export.py checkpoints/$DATASET.hpar checkpoints/$DATASET.ckpt --file_name mindir/$DATASET --device Ascend > logs/export_$DATASET.log 2>&1 &