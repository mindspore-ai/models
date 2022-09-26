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
  echo "         bash $0 OM_PATH DATA_PATH RESULT_PATH"
  echo "Example:"
  echo "         bash $0 ../models/CycleGAN_AtoB.om ../data/testA/ ./results"

  exit 255
fi

# build or rebuild
rm core
rm -r build
mkdir -p build
cd build || exit

function make_plugin() {
    if ! cmake ..;
    then
      echo "[ERROR] Cmake failed."
      return 1
    else
      echo "[INFO] Cmake done"
    fi

    if ! (make);
    then
      echo "[ERROR] Make failed."
      return 1
    else
      echo "[INFO] Make done"
    fi

    return 0
}

if make_plugin;
then
  echo "[INFO] Build successfully."
else
  echo "[ERROR] Build failed."
fi

cd - || exit

# run
echo "[INFO] Running"
rm -r results
mkdir -p results
OM_PATH=$1
DATA_PATH=$2
RESULT_PATH=$3
./build/CycleGAN "${OM_PATH}" "${DATA_PATH}" "${RESULT_PATH}"