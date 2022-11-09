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
    echo "bash run_eval.sh DATA_PATH CKPT_PATH CATEGORY DEVICE_ID"
    echo "For example: bash run_eval_gpu.sh /path/dataset /path/ckpt_path category 0"
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
CKPT_PATH=$(get_real_path $2)


eval_path=eval_$3
if [ -d $eval_path ];
then
    rm -rf ./$eval_path
fi
mkdir ./$eval_path
env > $eval_path/env.log
echo "[INFO] start eval dataset $4."
nohup python3 -u ../eval.py \
       --dataset_path "$DATA_PATH" \
       --ckpt_path "$CKPT_PATH" \
       --category "$3"  \
       --device_id "$4" \
       --device_target "GPU" \
        --test_batch_size 1 \
       > "$eval_path"/eval.log 2>&1 &

if [ $? -eq 0 ];then
    echo "[INFO] eval success"
else
    echo "[ERROR] eval failed"
    exit 2
fi
echo "[INFO] finish"
