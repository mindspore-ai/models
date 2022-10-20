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

if [ $# -eq 2 ]
then
    echo "Usage: bash ./scripts/run_eval_onnx.sh [DEVICE_ID] [CONFIG_PATH]"
exit 1
fi

export DEVICE_ID=$1
export CONFIG_PATH=$2

python eval_onnx.py --device_id=$DEVICE_ID \
    --ddr_config=$CONFIG_PATH \
    --file_format="ONNX" \
    --device_target="GPU"
