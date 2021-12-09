#!/bin/bash

# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export ASCEND_HOME=/usr/local/Ascend
export ARCH_PATTERN=x86_64-linux
export ASCEND_VERSION=nnrt/latest

# OM_FILE=/home/sjtu_liu/mindx/model/ssd_ghostnet/ssd_ghostnet.om

rm -rf dist
mkdir dist
cd dist
cmake ..
make -j
make install

cp ./ssd_ghost ../

# ./ssd_ghost ${OM_FILE} ./test.jpg