#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

echo "====================================================================================================================="
echo "Run this script as: "
echo "bash run_train.sh [device_target] [device_id]"
echo "for example: bash run_train.sh GPU 0"
echo "====================================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: bash run_train.sh [DEVICE_TARGET] [DEVICE_ID]"
    exit 1
fi

rm -rf ./train
mkdir ./train
mkdir ./train/results
mkdir ./train/results/fake_img
mkdir ./train/results/loss_show
mkdir ./train/results/ckpt
mkdir ./train/results/predict
cp ../*.py ./train
cp ../scripts/*.sh ./train
cp -r ../src ./train
cp -r ../*.yaml ./train
cd ./train || exit

python train.py --device_target=$1 --device_id=$2 &> log &
