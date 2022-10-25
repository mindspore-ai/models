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

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [CONFIG_PATH] [DEVICE_ID]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
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

if [ "$2" == "y" ] || [ "$2" == "n" ];then
    need_preprocess=$2
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

config_path=$3

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi


echo "mindir name: "$model
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d process_Result ]; then
        rm -rf ./process_Result
    fi
    mkdir process_Result
    python ../preprocess.py --cfg_path $config_path --ckpt_path $model --device_id=$device_id  --device_target Ascend
}

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    # voxels, num_points, coors, bev_map
    ../ascend310_infer/out/main --mindir_path=$model --input0_path=./process_Result/voxels_data --input1_path=./process_Result/num_points_data --input2_path=./process_Result/coors_data  --device_id=0 &> infer.log
}

function cal_acc()
{
    python ../postprocess.py  --label_file=./process_Result/kitti_infos_val.pkl --result_path=result_Files/  --input_data_path=./process_Result/ --cfg_path=$config_path &> acc.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
