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
if [ $# != 4 ]; then
  echo "Usage: bash eval.sh [DEVICE_ID] [CKPT_PATH] [DATA_PATH] [EVAL_SPLIT_FILE_PATH]"
  exit 1
fi

export DEVICE_ID=$1

python3 eval.py \
          --device_target Ascend \
          --dev_id "${DEVICE_ID}" \
          --ckpt_path $2\
          --data_path $3\
          --eval_split_file_path $4 > eval.log 2>&1 &

echo "run standalone eval on device ${DEVICE_ID}"
