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

if [ $# != 2 ] && [ $# != 3 ]; then
    echo "Usage: bash scripts/run_eval_gpu.sh [DATA_PATH] [CKPT_PATH] [NUM_CHECKPOINTS](optional)"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)
CKPT_PATH=$(get_real_path $2)

if [ ! -d $DATA_PATH ]; then
    echo "error: DATA_PATH=$DATA_PATH is not a dir"
exit 1
fi

if [ ! -d $CKPT_PATH ]; then
    echo "error: CKPT_PATH=$CKPT_PATH is not a dir"
exit 1
fi

if [ -d "scripts/eval" ];
then
    rm -rf scripts/eval
fi

mkdir scripts/eval
cp eval.py default_config.yaml scripts/run_eval_gpu.sh scripts/eval
cp -r src scripts/eval
cd scripts/eval || exit
echo "start eval"

if [ $# == 2 ]; then
    python eval.py --device_target GPU --data_path $DATA_PATH --ckpt_path $CKPT_PATH > log 2>&1 &
fi

if [ $# == 3 ]; then
    python eval.py --device_target GPU --data_path $DATA_PATH --num_to_eval $3 --ckpt_path $CKPT_PATH > log 2>&1 &
fi

cd ..
