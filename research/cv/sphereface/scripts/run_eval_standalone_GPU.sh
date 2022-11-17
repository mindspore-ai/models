#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_standalone_GPU.sh [DEVICE_ID] [CKPT_FILES]"
echo "=============================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: sh run_eval_standalone_GPU.sh [DEVICE_ID] [CKPT_FILES]"
    exit 1
fi

export DEVICE_ID=$1
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "./eval_GPU" ];
then
    rm -rf ./eval_GPU
fi
mkdir ./eval_GPU
cd ./eval_GPU || exit
mkdir src
cd ../
cp ../*.py ./eval_GPU
cp ../*.yaml ./eval_GPU
cp -r ../src ./eval_GPU/
cd ./eval_GPU

nohup python ${BASEPATH}/../eval.py \
    --is_distributed=0 \
    --device_target='GPU' \
    --device_id=$1 \
    --ckpt_files=$2 > eval.log 2>&1 &
