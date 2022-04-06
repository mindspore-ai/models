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

if [ $# != 1 ]
then
    echo "Usage: bash run_sdk_infer.sh [EVAL_DATA_DIR]"
    exit 1
fi

if [ ! -d $1 ]
then
    echo "error: EVAL_DATA_DIR=$1 is not a directory"
    echo "Usage: sh run_sdk_infer.sh [EVAL_DATA_DIR]"
exit 1
fi

echo "get dataset"

rm -rf ./310_eval_dataset/

python3.7 ../../get_310_eval_dataset.py --eval_data_dir $1

cp ../../src/get_text.py ./310_eval_dataset/

cd ./310_eval_dataset/

python3.7 get_text.py

cp ./infer.txt ../

echo "done"



echo "infer..."

cd ../

python3.7 main.py

echo "done"


echo "cal acc..."

python3.7 postprocess.py --eval_data_dir $1


