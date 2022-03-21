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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: run_standalone_eval_gpu.sh [TEST_DATA_PATH] [CKPT_PATH] [DEVICE_ID] [TEST_MODEL]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)

if [ ! -d $PATH1 ]
then
    echo "error: TEST_DATA_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $2 ]
then
    echo "error: CKPT_PATH=$2 is not a file"
exit 1
fi

echo "start evaluating for $TEST_MODEL"
python ../eval.py --data_path=$DATA_PATH --ckpt_path=$CKPT_PATH --device_target=GPU --device_id=$DEVICE_ID --test_model=$TEST_MODEL > eval_log 2>&1 &
