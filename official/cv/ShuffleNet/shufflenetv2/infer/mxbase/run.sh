#!/bin/bash

# Copyright (c) 2021. Huawei Technologies Co., Ltd.
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
if [ $# != 1 ]
then
    echo "Usage: bash run.sh [DATASET_VAL_PATH]"
exit 1
fi

# check the DATASET_VAL_PATH
if [ ! -d $1 ]
then
    echo "error: DATASET_VAL_PATH=$1 is not a path"
exit 1
fi

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}

# run
./shufflenetv2 $1
