#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [[ $# -lt 3 || $# -gt 7 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [KEEP_RATIO](optional) [DEVICE_ID](optional)
    IMAGE_WIDTH, IMAGE_HEIGHT and DEVICE_ID is optional. IMAGE_WIDTH and IMAGE_HEIGHT must be set at the same time or not at the same time. 
    IMAGE_WIDTH default value is 1280, IMAGE_HEIGHT default value is 768, KEEP_RATIO default value is true, must be the same as that during 
    training. DEVICE_ID can be set by environment variable device_id, otherwise the value is zero"
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
data_path=$(get_real_path $2)
anno_path=$(get_real_path $3)
device_id=0
image_width=1280
image_height=768
# If keep_ratio is set to False during model training, it should be also set to false here.
keep_ratio=true
# If restore_bbox is set to False during export.py, it should be also set to false here (only support true or false and case sensitive).
restore_bbox=true

if [ $# -eq 4 ]; then
    device_id=$4
fi

if [ $# -eq 5 ]; then
    image_width=$4
    image_height=$5
fi

if [ $# -eq 6 ]; then
    image_width=$4
    image_height=$5
    keep_ratio=$6
fi
if [ $# -eq 7 ]; then
    image_width=$4
    image_height=$5
    keep_ratio=$6
    device_id=$7
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "anno_path: " $anno_path
echo "device id: "$device_id
echo "image_width: "$image_width
echo "image_height: "$image_height
echo "keep_ratio: "$keep_ratio
echo "restore_bbox: "$restore_bbox

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
else
    export ASCEND_HOME=/usr/local/Ascend/latest
fi
export PATH=$ASCEND_HOME/compiler/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=$ASCEND_HOME/opp

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    cd - || exit
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=$data_path --device_id=$device_id --IMAGEWIDTH=$image_width --IMAGEHEIGHT=$image_height  --KEEP_RATIO=$keep_ratio --RESTOREBBOX=$restore_bbox &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --anno_path=$anno_path --result_path=./result_Files &> acc.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi