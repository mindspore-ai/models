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

for para in "$@"
do
    if [[ $para == --model* ]];then
        model=`echo ${para#*=}`
    elif [[ $para == --output* ]];then
        output=`echo ${para#*=}`
    elif [[ $para == --soc_version* ]];then
        soc_version=`echo ${para#*=}`
    fi
done

echo "Input AIR file path: ${model}"
echo "Input aipp file path: ${output}"

soc_version=Ascend310

atc --input_format=NCHW \
    --model=${model} \
    --output=${output} \
    --soc_version=${soc_version} \
    --framework=1
