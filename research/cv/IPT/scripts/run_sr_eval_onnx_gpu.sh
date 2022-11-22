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
if [ $# -ne 4 ]; then
    echo "Usage: bash run_eval_onnx_gpu.sh [ONNX_FILE_PATH] [DATASET] [DATA_PATH] [SCALE]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
data_path="$(get_real_path $3)"

if [ $# == 4 ]
then
    if [ $4 -eq 2 ]; then
        python ${BASE_PATH}/../eval_onnx.py --dir_data $data_path --data_test $2 --pth_path $1 --test_only --ext img --scale 2 --task_id 0 > eval_onnx.log 2>&1 &
    elif [ $4 -eq 3 ]; then
        python ${BASE_PATH}/../eval_onnx.py --dir_data $data_path --data_test $2 --pth_path $1 --test_only --ext img --scale 3 --task_id 0 > eval_onnx.log 2>&1 &
    elif [ $4 -eq 4 ]; then
        python ${BASE_PATH}/../eval_onnx.py --dir_data $data_path --data_test $2 --pth_path $1 --test_only --ext img --scale 4 --task_id 0 > eval_onnx.log 2>&1 &
    else
        echo "error: the selected dataset is not in supported set{set5, set14, B100, Urban100, CBSD68, Rain100L}"
    exit 1
    fi
fi
