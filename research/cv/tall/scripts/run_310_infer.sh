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
# limitations under the License.
# ============================================================================

if [ $# != 3 ]
then
    echo "Usage: bash run_310_infer.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID]."
    exit 1
fi

if [ ! -f $1 ]
then
    echo "error: MINDIR_PATH=$1 is not a file"
    echo "Usage: sh run_eval.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID]."
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: EVAL_DATA_DIR=$2 is not a directory"
    echo "Usage: sh run_eval.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID]."
exit 1
fi


echo "get dataset"

python ../get_310_eval_dataset.py --eval_data_dir $2

echo "done"


echo "compile..."

cd ../ascend310_infer || exit
bash build.sh &> build.log
cd build 
make
cd ../../scripts

echo "compile done"


echo "infer..."

if [ -d result_Files ];
then
    rm -rf result_Files
fi

if [ -d time_Result ];
then
    rm -rf time_Result
fi

mkdir result_Files
mkdir time_Result

input_path="./310_eval_dataset"
../ascend310_infer/build/main $1 $input_path  $3


echo "cal acc..."
python ../postprocess.py --eval_data_dir $2 --device_id $3 > acc.log 2>&1 &



