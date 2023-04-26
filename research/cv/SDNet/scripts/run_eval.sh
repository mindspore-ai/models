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

ulimit -u unlimited

if [ $# != 2 ]
then
    echo "Usage: bash run_eval.sh [DATA_PATH] [PATH_CHECKPOINT]"
exit 1
fi

if [ ! -d $1 ]
then
    echo "error: DATA_PATH=$1 is not a directory"
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
export DEVICE_ID=0

python ${BASEPATH}/../eval.py --dataroot=$1 --checkpoint_path=$2 > ./eval.log 2>&1 &
