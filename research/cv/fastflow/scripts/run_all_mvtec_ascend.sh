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

if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_all_mvtec_gpu.sh DATA_PATH PRETRAINED_PATH DEVICE_ID"
    echo "For example: bash run_all_mvtec_ascend.sh /path/dataset /path/pretrained_path 0"
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
PRE_CKPT_PATH=$(get_real_path $2)

train_path="train_all"
if [ -d "$train_path" ];
then
    rm -rf ./"$train_path"
fi
mkdir ./"$train_path"

eval_path="eval_all"
if [ -d "$eval_path" ];
then
    rm -rf ./"$eval_path"
fi
mkdir ./"$eval_path"

arr=("bottle" "cable" "capsule" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

for value in "${arr[@]}"
do
  python3 -u ../train.py \
      --dataset_path "$DATA_PATH" \
      --pre_ckpt_path "$PRE_CKPT_PATH" \
      --category "$value" \
      --device_id "$3" \
      --device_target "Ascend" \
      > "$train_path"/train_$value.log

  python3 -u ../eval.py \
      --dataset_path "$DATA_PATH" \
      --ckpt_path "" \
      --category "$value" \
      --device_id "$3" \
      --device_target "Ascend" \
      --test_batch_size 1 \
      > "$eval_path"/eval_$value.log
done

pixel_auc=$(grep "auc" eval_all/*.log | awk -F "pixel_auc:" '{print $2}' | awk '{sum+=$1}END{print sum/NR}' | awk '{printf("%.4f", $1)}')
echo "[INFO] average pixel_auc = $pixel_auc"
