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


# env
function check_env()
{
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
    if [ ! "${ASCEND_HOME}" ]; then
        export ASCEND_HOME=/usr/local/Ascend/
        echo "Set ASCEND_HOME to the default value: ${ASCEND_HOME}"
    else
        echo "ASCEND_HOME is set to ${ASCEND_HOME} by user"
    fi

    if [ ! "${ASCEND_VERSION}" ]; then
        export ASCEND_VERSION=nnrt/latest
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        export ARCH_PATTERN=./
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi

    export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}
}

function build()
{
    rm -rf build
    mkdir -p build
    cd build || exit

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

    cd - || exit
    return 0
}

check_env

if build;
then
  echo "INFO: Build successfully."
else
  echo "ERROR: Build failed."
fi
