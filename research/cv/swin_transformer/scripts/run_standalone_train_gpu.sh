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
if [ $# -lt 2 ]
then
    echo "Usage: bash ./scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]"
exit 1
fi
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export CONFIG_PATH=$1
export CUDA_VISIBLE_DEVICES="$2"
export RANK_SIZE=1
export DEVICE_NUM=1
export DEPLOY_MODE=0
# export LD_LIBRARY_PATH="/usr/local/cuda-11.1/extras/CUPTI/lib64"
export GE_USE_STATIC_MEMORY=1
rm -rf train_gpu_alone
mkdir ./train_gpu_alone
cd ./train_gpu_alone || exit
env > env.log
# pip show mindspore_gpu
# python -c "import mindspore;mindspore.run_check()"
nohup python ${BASEPATH}/../train.py  --device_target="GPU" \
    --swin_config $CONFIG_PATH \
    --start_epoch 0 \
    --epochs 350 > log.txt 2>&1 &
cd ../


