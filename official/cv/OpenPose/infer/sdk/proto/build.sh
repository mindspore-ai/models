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
# limitations under the License.mitations under the License.

set -e 

current_folder="$( cd "$(dirname "$0")" ;pwd -P )"

function build_plugin() {
    build_path=$current_folder/build
    if [ -d "$build_path" ]; then
        rm -rf "$build_path"
    else
        echo "file $build_path is not exist."
    fi
    mkdir -p "$build_path"
    cd "$build_path"
    cmake ..
    make -j
    cd ..
    exit 0
}

build_plugin
exit 0
