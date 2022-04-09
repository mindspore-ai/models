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

if [ $# -lt 4 ]
then
    echo "Usage: \
          bash run_eval_gpu.sh [MpiSintelClean/MpiSintelFinal] [DATA_PATH] [MODEL_NAME] [CKPT_PATH] [DEVICE_ID]\
          "
exit 1
fi


export DATA_NAME=$1
export DATA_PATH=$2
export MODEL_NAME=$3
export CKPT_PATH=$4
export DEVICE_ID=$5

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

ulimit -u unlimited

CONFIG_PATH="${BASEPATH}/../default_config.yaml"
echo "config path is : ${CONFIG_PATH}"


python3 eval.py --config_path=$CONFIG_PATH --eval_data=$DATA_NAME \
    --eval_data_path=$DATA_PATH --model=$MODEL_NAME --eval_checkpoint_path=$CKPT_PATH \
    --device_id=$DEVICE_ID --device_target="GPU" > eval.log 2>&1 &
