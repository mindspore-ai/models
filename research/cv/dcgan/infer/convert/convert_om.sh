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

# Parameter format
if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 INPUT_AIR_PATH OUTPUT_OM_PATH_NAME"
  echo "Example:"
  echo "         bash $0 ./dcgan_16_20220106.air ../models/DCGAN"

  exit 255
fi

# DCGAN model from .air to .om
AIR_PATH=$1
OM_PATH=$2
atc --input_format=NCHW \
--framework=1 \
--model="${AIR_PATH}" \
--output="${OM_PATH}" \
--soc_version=Ascend310

# Delete unnecessary files
rm fusion_result.json
rm -r kernel_meta/

# Modify file permissions
chmod +r+w "${OM_PATH}.om"
