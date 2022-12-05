#!/usr/bin/env bash

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

model_path=$1
output_model_name=$2

echo "Input AIR file path: ${model_path}"
echo "Output OM file path: ${output_model_name}"

atc --model=$model_path/rotate-head.air\
        --framework=1 \
        --output=$output_model_name-head \
        --input_format=NCHW \
        --soc_version=Ascend310

atc --model=$model_path/rotate-tail.air \
        --framework=1 \
        --output=$output_model_name-tail \
        --input_format=NCHW \
        --soc_version=Ascend310