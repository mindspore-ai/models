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

if [ $# != 4 ]
then
    echo "Usage: bash run_infer_onnx.sh [DEVICE_TARGET] [DEVICE_ID] [ONNX_INFER_DATA_DIR] [ONNX_PATH]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$3)"
  fi
}

PATH1=$(get_real_path $3)
ONNX_PATH=$(get_real_path $4)
if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi

python infer_onnx.py --device_target=$1 --device_id=$2 --onnx_infer_data_dir=$PATH1 --onnx_path=$ONNX_PATH &> infer_onnx.log 2>&1 &
