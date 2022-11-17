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

if [ $# != 4 ]; then
  echo "Usage: 
        bash run_eval_ascend.sh [config_file] [test_dataset_path] [pretrain_path] [ckpt_name]
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

CONFIG=$(get_real_path $1)
echo "CONFIG: "$CONFIG

DATASET=$(get_real_path $2)
echo "DATASET: "$DATASET

pretrain_path=$(get_real_path $3)
echo "pretrain_path: "$pretrain_path

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -d $DATASET ]
then
    echo "error: dataset=$DATASET is not a directory."
exit 1
fi

if [ ! -d $pretrain_path ]
then
    echo "error: pretrain_path=$pretrain_path is not a directory."
exit 1
fi

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

echo "Evaluating on Ascend..."
echo
env > env.log
pwd
echo
python ${BASE_PATH}/../eval.py --device_target=Ascend --config_path=$CONFIG --test_dir=$DATASET \
  --pretrain_path=$pretrain_path --ckpt_name=$4 &> eval.log &