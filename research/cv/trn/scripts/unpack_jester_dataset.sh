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

if [ $# != 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/unpack_jester_dataset.sh [DATA_PATH] [TARGET_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

DATA_PATH=$(get_real_path "$1")
TARGET_PATH=$(get_real_path "$2")

# Check the specified dataset root directory
if [ ! -d "$DATA_PATH" ]
then
  echo "The specified dataset root is not a directory: \"$DATA_PATH\"."
  exit 1
fi

echo "Target data folder: $TARGET_PATH"
mkdir -p "$TARGET_PATH"

echo "Unzip..."
unzip "$DATA_PATH/20bn-jester-download-package-labels.zip" -d "$TARGET_PATH"
unzip "$DATA_PATH/20bn-jester-v1-??.zip" -d "$TARGET_PATH"

cd "$TARGET_PATH" || exit 1
echo "Extract tar archives ..."
cat ./20bn-jester-v1-?? | tar zx -v

echo "Remove unnecessary data..."
rm ./20bn-jester-v1-??.md5
rm ./20bn-jester-v1-??

echo "DONE!"
