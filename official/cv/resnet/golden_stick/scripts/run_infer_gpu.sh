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

CURPATH="$(dirname "$0")"

if [ $# != 4 ]
then
    echo "Usage: bash run_eval_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [MINDIR_PATH]"
    echo "PYTHON_PATH represents path to directory of 'infer.py'."
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PYTHON_PATH=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
DATASET_PATH=$(get_real_path $3)
MINDIR_FILE=$(get_real_path $4)


if [ ! -d $PYTHON_PATH ]
then 
    echo "error: PYTHON_PATH=$PYTHON_PATH is not a directory"
exit 1
fi 

if [ ! -f $CONFIG_FILE ]
then 
    echo "error: CONFIG_FILE=$CONFIG_FILE is not a file"
exit 1
fi

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f $MINDIR_FILE ]
then
    echo "error: MINDIR_FILE=$MINDIR_FILE is not a file"
exit 1
fi

ulimit -u unlimited

if [ -d "infer" ];
then
    rm -rf ./infer
fi
mkdir ./infer
cp ${PYTHON_PATH}/*.py ./infer
cp -r ${CURPATH}/../../src ./infer
cd ./infer || exit
env > env.log
echo "start infer"
python infer.py --config_path=$CONFIG_FILE --data_path=$DATASET_PATH --mindir_path=$MINDIR_FILE  \
      --device_target="GPU" &> log &
cd ..
