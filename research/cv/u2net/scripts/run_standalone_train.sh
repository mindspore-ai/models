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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_train.sh CONTENT_PATH LABEL_PATH DEVICE_TARGET"
echo "for example: bash scripts/run_standalone_train.sh /path/to/content /path/to/label GPU"
echo "=============================================================================================================="

if [ ! -d $1 ]
then
    echo "error: CONTENT_PATH=$1 is not a directory"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: LABEL_PATH=$2 is not a directory"
exit 1
fi

if [ $3 != "Ascend" ] && [ $3 != "GPU" ]
then
    echo "error: DEVICE_TARGET should be Ascend or GPU, but got $3"
exit 1
fi

python train.py  --run_distribute 0 --content_path $1 --label_path $2 --device_target $3 > output.log 2>&1 &
