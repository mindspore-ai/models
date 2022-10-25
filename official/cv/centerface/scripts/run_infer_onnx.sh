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

if [ $# != 4 ]
then
    echo "Usage: bash run_infer_onnx.sh [DEVICE_ID] [ONNX] [DATASET] [GROUND_TRUTH_MAT]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

current_exec_path=$(pwd)
dirname_path=$(dirname "$(pwd)")

SCRIPT_NAME='infer_onnx.py'

ulimit -c unlimited

if [ $1 -lt 0 ] && [ $1 -gt 7 ]
then
    echo "error: DEVICE_ID=$1 is not in (0-7)"
    exit 1
fi

device_id=$1
export CUDA_VISIBLE_DEVICES="$1"

root=${current_exec_path} # your script path
save_path=$root/output/centerface/888
echo "save_path: "$save_path

onnx_path=$(get_real_path $2)
if [ ! -f $onnx_path ]
then
    echo "error: onnx_path=$onnx_path is not a file"
exit 1
fi

onnx_dir=$(dirname $onnx_path)
echo "onnx_path: "$onnx_path

dataset_path=$(get_real_path $3)
if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a dir"
exit 1
fi

ground_truth_mat=$(get_real_path $4)
if [ ! -f $ground_truth_mat ]
then
    echo "error: ground_truth_mat=$ground_truth_mat is not a file"
exit 1
fi

ground_truth_path=$(dirname $ground_truth_mat)

echo "dataset_path: "$dataset_path
echo "ground_truth_mat: "$ground_truth_mat
echo "ground_truth_path: "$ground_truth_path
echo "save_path: "$save_path

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1

echo 'start testing'
rm -rf ${current_exec_path}/device_test$device_id
rm -rf $save_path
echo 'start rank '$device_id
mkdir ${current_exec_path}/device_test$device_id
mkdir -p $save_path
cd ${current_exec_path}/device_test$device_id || exit
export RANK_ID=0
dev=`expr $device_id + 0`
export DEVICE_ID=$dev
python ${dirname_path}/${SCRIPT_NAME} \
    --is_distributed=0 \
    --data_dir=$dataset_path \
    --test_model=$onnx_dir \
    --ground_truth_mat=$ground_truth_mat \
    --save_dir=$save_path \
    --rank=$device_id \
    --onnx_path=$onnx_path \
    --device_target="GPU" \
    --eval=1 \
    --ground_truth_path=$ground_truth_path > test_onnx.log  2>&1 &

echo 'running'
