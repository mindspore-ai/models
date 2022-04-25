#!/bin/bash
# Copyright (c) 2022. Huawei Technologies Co., Ltd
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

if [ $# -ne 2 ]
then
    echo "Need two parameters: one for air model input file path, another for om model output dir path!"
    exit 1
fi

model=$1
output=$2

atc --model="${model}" \
    --framework=1 \
    --output="${output}" \
    --soc_version=Ascend310 \
    --input_shape="data:1,499" \
    --output_type=FP16