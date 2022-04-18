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

export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}

mkdir -p build
cd build || exit

function make_plugin() {
    if ! cmake ..;
    then
      echo "cmake failed."
      return 1
    fi

    if ! (make);
    then
      echo "make failed."
      return 1
    fi

    return 0
}

if make_plugin;
then
  echo "INFO: Build successfully."
else
  echo "ERROR: Build failed."
fi

cd - || exit
