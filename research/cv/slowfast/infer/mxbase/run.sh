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
    echo "Usage: bash run.sh [DATA_DIR] [OM_PATH]"
    exit 1
fi

if [ ! -d $1 ]
then
    echo "error: DATA_DIR=$1 is not a directory"
    echo "Usage: bash run.sh [DATA_DIR] [OM_PATH]"
exit 1
fi

if [ ! -f $2 ]
then
    echo "error: OM_PATH=$2 is not a file"
    echo "Usage: bash run.sh [EVAL_DATA_DIR] [OM_PATH]"
exit 1
fi

echo "compile..."

bash build.sh

echo "done"

echo "infer..."

data_dir=$1
om_path=$2

./slowfast  ${data_dir} ${om_path}

echo "done"

echo "cal acc..."
cp -r ../utils/* ./
cp -r ../../src ./
python3 CalPrecision.py --data_dir ${data_dir}

echo "done"
