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
# ============================================================================

input_air_path=$1
output_om_path=$2
aipp_cfg=$3


echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"
echo "AIPP cfg file path: ${aipp_cfg}"

atc  --input_format=NCHW --framework=1 \
--model=${input_air_path} \
--output=${output_om_path} \
--soc_version=Ascend310 \
--disable_reuse_memory=0 \
--insert_op_conf=${aipp_cfg} \
--precision_mode=allow_fp32_to_fp16  \
--op_select_implmode=high_precision