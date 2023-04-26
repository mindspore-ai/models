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

if [ $# -ne 2 ]
then
    echo "Usage: bash run_eval.sh [CHECKPOINT_FILE_PATH] [TEST_DIR]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1
if [ ! -f $PATH1 ]
then
    echo "error: checkpoint_file_path=$PATH1 is not a file"
exit 1
fi

PATH2=$(get_real_path $2)
echo $PATH2
if [ ! -d $PATH2 ]
then
    echo "error: eval_dir=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

rm -rf ./test
mkdir ./test
cp ../*.py ./test
cp *.sh ./test
cp ../*.yaml ./test
cp -r ../src ./test
cd ./test || exit
echo "start test"
env > env.log
python test.py --checkpoint_file_path=$PATH1 --test_dir=$PATH2 --output_path './output' > test.log 2>&1 &
cd ..