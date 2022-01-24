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
if [ $# != 0 ] && [ $# != 1 ]
then
    echo "Usage: bash run_eval.sh [DEVICE_ID]"
exit 1
fi

DEVID=0
if [ $# == 1 ]; then
    DEVID=$1
fi
export DEVICE_ID=$DEVID
echo "device is: ${DEVID}"

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

config_path="${BASEPATH}/../config.yaml"
config_path=$(realpath $config_path)
echo "config path is : ${config_path}"

eval_script="${BASEPATH}/../eval.py"
eval_script=$(realpath $eval_script)
echo "eval.py path is : ${eval_script}"

cd ../
echo "current work path is : $(pwd)"

python3 ${eval_script} --yaml_path=$config_path  > ./scripts/eval.log 2>&1 &