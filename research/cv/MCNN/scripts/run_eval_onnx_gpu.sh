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
    echo "Usage: bash run_eval_onnx_gpu.sh [VAL_PATH] [VAL_GT_PATH] [ONNX_PATH]"
exit 1
fi

ulimit -u unlimited
export DEVICE_ID=0
export RANK_SIZE=1
export VAL_PATH=$1
export VAL_GT_PATH=$2
export ONNX_PATH=$3

if [ -d "eval_onnx" ];
then
    rm -rf ./eval_onnx
fi

mkdir ./eval_onnx
cp ../*.py ./eval_onnx
cp *.sh ./eval_onnx
cp -r ../src ./eval_onnx
cd ./eval_onnx || exit
env > env_onnx.log
echo "start evaluation for device $DEVICE_ID"
python eval_onnx.py --val_path=$VAL_PATH \
                    --val_gt_path=$VAL_GT_PATH --onnx_path=$ONNX_PATH &> log_onnx &
cd ..
