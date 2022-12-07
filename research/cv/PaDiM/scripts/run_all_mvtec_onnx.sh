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
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_all_mvtec_onnx.sh DATA_PATH PRETRAINED_PATH SAVE_PATH DEVICE_ID"
    echo "For example: bash run_all_mvtec_onnx.sh /path/dataset /path/pretrained_path /path/save_path 0"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
ONNX_PATH=$(get_real_path $2)
SAVE_PATH=$(get_real_path $3)
export DATA_PATH=$DATA_PATH

arr=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

for value in "${arr[@]}"
do
  bash run_train_onnx.sh  $DATA_PATH $ONNX_PATH $SAVE_PATH $value $4
  bash run_eval_onnx.sh  $DATA_PATH $ONNX_PATH $SAVE_PATH $value $4
done

img_auc=$(grep "auc" *_eval/eval.log | awk -F "img_auc:" '{print $2}' | awk -F "," '{print $1}' | awk '{sum+=$1}END{print sum/NR}' | awk '{printf("%.3f", $1)}')
echo "[INFO] average img_auc = $img_auc"

pixel_auc=$(grep "auc" *_eval/eval.log | awk -F "img_auc:" '{print $2}' | awk -F "pixel_auc:" '{print $2}' | awk '{sum+=$1}END{print sum/NR}' | awk '{printf("%.3f", $1)}')
echo "[INFO] average pixel_auc = $pixel_auc"
