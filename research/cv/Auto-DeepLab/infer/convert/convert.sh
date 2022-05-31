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
AIR_PATH=$1
OM_PATH=$2

echo "Input path of AIR file: ${AIR_PATH}"
echo "Output path of OM file: ${OM_PATH}"

atc --framework=1 \
    --model="${AIR_PATH}" \
    --output="${OM_PATH}" \
    --input_format=NHWC \
    --input_shape="actual_input_1:1,1024,2048,3" \
    --output_type=FP32 \
    --soc_version=Ascend310
exit 0
