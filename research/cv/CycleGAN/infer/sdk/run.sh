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
  echo "         bash $0 DATA_PATH RESULT_PATH"
  echo "Example:"
  echo "         bash $0 ../data/testA/ ./results"

  exit 255
fi

# Rebuild results folder
rm -r results
mkdir -p results

# Run main.py
DATA_PATH=$1
RESULT_PATH=$2
python3 main.py "${DATA_PATH}" "${RESULT_PATH}"
