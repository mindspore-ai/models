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
# ===========================================================================

if [ $# != 2 ]
then
    echo "Usage: bash run_eval.sh [CONFIG_PATH] [DEVICE_ID]"
exit 1
fi
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_PATH=$(get_real_path $1)
export DEVICE_ID=$2
EVAL_PATH=eval_`echo $CONFIG_PATH | rev | cut -d '/' -f 1 | rev | awk -F "_config.yaml" '{print $1}'`
if [ -d $EVAL_PATH ];
then
    rm -rf $EVAL_PATH
fi
mkdir $EVAL_PATH
cd $EVAL_PATH/ || exit
python ../eval.py --config_path=$CONFIG_PATH > eval.log 2>&1 &
tail -f eval.log
