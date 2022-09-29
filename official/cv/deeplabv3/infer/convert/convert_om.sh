#!/bin/bash

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air xx"

  exit 1
fi

model_path=$1
output_model_name=$2

atc --framework=1 \
    --input_format=NHWC \
    --model=$model_path \
    --output=$output_model_name \
    --fusion_switch_file=fusion_switch.cfg \
    --precision_mode=allow_fp32_to_fp16 \
    --op_select_implmode=high_precision \
    --soc_version=Ascend310
