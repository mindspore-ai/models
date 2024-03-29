#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
if [ $# != 1 ] && [ $# != 0 ]; then
  echo "Usage: bash run_train.sh [DEVICE_ID]"
  exit 1
fi

if [ $# == 1 ]; then
  export DEVICE_ID=$1
else
  export DEVICE_ID=0
fi

ulimit -u unlimited

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
python ${BASEPATH}/../train.py  > ./train.log 2>&1 &