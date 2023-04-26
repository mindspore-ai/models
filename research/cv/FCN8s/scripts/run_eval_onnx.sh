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


echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_eval.sh ONNX_PATH RANK_TABLE_FILE DATASET CONFIG_PATH "
echo "for example: bash scripts/run_eval.sh path/to/onnx_path /path/to/dataroot /path/to/dataset"
echo "It is better to use absolute path."
echo "================================================================================================================="
if [ $# -lt 3 ]; then
    echo "Usage: bash ./scripts/run_eval_onnx.sh [ONNX_PATH] [DATA_ROOT] [DATA_PATH]"
exit 1
fi
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
export ONNX_PATH=$(get_real_path $1)
export DATA_ROOT=$2
export DATA_PATH=$3
if [ ! -f $ONNX_PATH ]
then
    echo "error: ONNX_PATH=$ONNX_PATH is not a file"
exit 1
fi
rm -rf eval
mkdir ./eval
cp ./*.py ./eval
cp ./*.yaml ./eval
cp -r ./src ./eval
cd ./eval || exit
echo "start testing"
env > env.log
python eval_onnx.py  \
--onnx_path=$ONNX_PATH \
--data_root=$DATA_ROOT  \
--data_lst=$DATA_PATH   \
--device_target="GPU"  \
--eval_batch_size=1 #> log.txt 2>&1 &
