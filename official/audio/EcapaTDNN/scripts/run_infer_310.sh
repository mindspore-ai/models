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

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: sh run_infer_310.sh [MODEL_PATH] [DATA_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

model=$(get_real_path $1)
datapath=$(get_real_path $2)
if [ $# == 2 ]; then
    device_id=$2
elif [ $# == 1 ]; then
    if [ -z $device_id ]; then
        device_id=0
    else
        device_id=$device_id
    fi
fi

function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function infer()
{
    if [ -d testdata ]; then
        rm -rf ./testdata
    fi
    mkdir testdata
    python ../ascend310_infer/310infer_preprocess.py $datapath
    if [ -d output ]; then
        rm -rf ./output
    fi
    mkdir output
    output_path=$(get_real_path output)
    ../ascend310_infer/build/ecapa_sample $model testdata $output_path $device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}
compile_app
infer
src_path=`dirname $0 | xargs -i realpath ../{}`
export PYTHONPATH=$src_path:$PYTHONPATH
python ../ascend310_infer/eval_310.py > acc.log
