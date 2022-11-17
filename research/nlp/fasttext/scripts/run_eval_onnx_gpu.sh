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
echo "bash run_eval_onnx.sh DATASET_PATH DATASET_NAME ONNX_PATH"
echo "for example: bash run_eval_onnx.sh /home/workspace/ag/ ag device0/ckpt0"
echo "It is better to use absolute path."
echo "=============================================================================================================="
if [ $# != 3 ]
then
    echo "Usage: bash run_eval_onnx.sh [DATASET_PATH] [DATASET_NAME] [ONNX_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
echo $DATASET
DATANAME=$2
ONNX_PATH=$(get_real_path $3)
echo "ONNX_PATH:${ONNX_PATH}"
echo "DATANAME: ${DATANAME}"
config_path="./${DATANAME}_config.yaml"
echo "config path is : ${config_path}"

if [ -d "eval_onnx" ];
then
    rm -rf ./eval_onnx
fi
mkdir ./eval_onnx
cp ../*.py ./eval_onnx
cp ../*.yaml ./eval_onnx
cp -r ../src ./eval_onnx
cp -r ../model_utils ./eval_onnx
cp -r ../scripts/*.sh ./eval_onnx
cd ./eval_onnx || exit
echo "start eval on standalone GPU"

python ../../eval_onnx.py \
--config_path $config_path \
--device_target GPU \
--onnx_path $ONNX_PATH \
--dataset_path $DATASET \
--data_name $DATANAME > eval_onnx.log 2>&1 &
cd ..
