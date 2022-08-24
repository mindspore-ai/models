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
    echo "Usage: bash run_infer_onnx.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID] [ONNX_PATH]"
exit 1
fi


export DATA_PATH=$1
export EXPERIMENT_NAME=$2
export DEVICE_ID=$3
export ONNX_PATH=$4

python infer_stgan_onnx.py --dataroot=$DATA_PATH --experiment_name=$EXPERIMENT_NAME \
                    --device_id=$DEVICE_ID --onnx_path=$ONNX_PATH --platform="GPU" > infer_onnx_log 2>&1 &