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

if [ $# != 2 ]; then
  echo "Usage:
        bash run_eval_onnx_gpu.sh [ONNX_NAME] [TESTSET_PATH]
        "
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

ONNX_NAME=$(get_real_path $1)
echo "ONNX_NAME: "$ONNX_NAME

TESTSET_PATH=$(get_real_path $2)
echo "TESTSET_PATH: "$TESTSET_PATH


if [ ! -f $ONNX_NAME ]
then
    echo "error: ONNX_NAME=$ONNX_NAME is not a file."
exit 1
fi

if [ ! -d $TESTSET_PATH ]
then
    echo "error: TESTSET_PATH=$TESTSET_PATH is not a directory."
exit 1
fi

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

echo "Evaluating on GPU..."
echo
env > env.log
pwd
echo
python ${BASE_PATH}/../eval_onnx.py --onnx_name=$ONNX_NAME --test_dir=$TESTSET_PATH --device_target=GPU &> eval.log &
