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
if [ $# != 4 ] && [ $# != 5 ]
then
    echo "Usage: bash run_export_gpu.sh [CFG_PATH] [CKPT_PATH] [FILE_NAME] [DEVICE_ID] [FILE_FORMAT](optional)"
exit 1
fi

CFG_PATH=$1
CKPT_PATH=$2
FILE_NAME=$3

export DEVICE_ID=$4

if [ $# == 5 ]
then
    FILE_FORMAT="$5"
else
    FILE_FORMAT="MINDIR"
fi

python export.py \
  --cfg_path="$CFG_PATH" \
  --ckpt_path="$CKPT_PATH" \
  --file_name="$FILE_NAME" \
  --file_format="$FILE_FORMAT"
