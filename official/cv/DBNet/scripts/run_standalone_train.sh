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
if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID] [LOG_NAME](optional)"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

CONFIG_PATH=$(get_real_path $1)
LOG_NAME="standalone_train"
if [ $# == 3 ]
then
    LOG_NAME=$3
fi
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
cd $BASE_PATH/..
python train.py --config_path=$CONFIG_PATH --device_id=$2 --output_dir=$LOG_NAME > $LOG_NAME.txt 2>&1 &
echo "training..."
echo "log at ${LOG_NAME}.txt, you can use [tail -f ${LOG_NAME}.txt] to get log."
