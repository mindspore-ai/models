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

if [ $# -lt 1 ]
then
  echo "Usage: bash run_standalone_train.sh DEVICE_ID"
  exit 1
fi

ulimit -u unlimited

export DEVICE_ID=$1
export RANK_ID=0

rm -rf ./train_parallel_standalone
mkdir ./train_parallel_standalone
cp ../*.py ./train_parallel_standalone
cp *.sh ./train_parallel_standalone
cp -r ../weights ./train_parallel_standalone
cp -r ../src ./train_parallel_standalone
cd ./train_parallel_standalone || exit

env > env.log

nohup python train.py \
            --device_num=1 \
            --rank_id=$RANK_ID \
            --distribute=False > log 2>&1 &

cd ..
