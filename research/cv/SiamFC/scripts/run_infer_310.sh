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

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MODEL_PATH1] [MODEL_PATH2] [SEQ_ROOT_PATH] [CODE_PATH] [DEVICE_TARGET] [DEVICE_ID]
    DEVICE_TARGET must choose from ['GPU', 'CPU', 'Ascend']
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path()
{
  if [ -z "$1" ]; then
       echo ""
  elif [ "${1:0:1}" == "/" ]; then
       echo "$1"
  else
       echo "$(realpath -m $PWD/$1)"
  fi
}

model1=$(get_real_path $1)
model2=$(get_real_path $2)
seq_root_path=$(get_real_path $3)
code_path=$(get_real_path $4)

if [ "$5" == "GPU" ] || [ "$5" == "CPU" ] || [ "$5" == "Ascend" ];then
    device_target=$5
else
  echo "device_target must be in  ['GPU', 'CPU', 'Ascend']"
  exit 1
fi

device_id=0
if [ $# == 4 ]; then
    device_id=$6
fi

echo "mindir name1: "$model1
echo "mindir name2: "$model2
echo "dataset path: "$seq_root_path
echo "device_target: "$5
echo "device id: "$6
echo "code path: "$code_path

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d ../results/OTB2013/SiamFC/times ]; then
        rm -rf ../results
    fi

    mkdir -p ../results/OTB2013/SiamFC/times
    ../ascend310_infer/out/main --model_path1=$model1 --model_path2=$model2  --device_id=$device_id --device_target=$device_target --seq_root_path=$seq_root_path --code_path=$code_path &>infer.log

}

function cal_acc()
{
     cd $code_path
     python postprocess.py --device_id=$device_id --dataset_path=$seq_root_path &> ./scripts/acc.log
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"   
fi
cal_acc


