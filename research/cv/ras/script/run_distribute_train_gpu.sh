#!/usr/bin/env bash

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



echo "===================================================================================================="
echo "Please run the script as:"
echo "bash script/run_distribute_train_gpu.sh [RANK_SIZE] [data_url] [pre_model] [train_url]"
echo "for example: bash script/run_distribute_train_gpu.sh 8 /home/data/ /home/resent50.ckpt /home/data/models/"
echo " *****
      RANK_SIZE: The parameter is numbers of device for distributed training;
      data_url: The data_url directory is the directory where the data set is located,and there must be two folders, images and labels, under data_url;
      pre_model: path of pretrained model;
      train_url: the save path of checkpoint file."
echo "===================================================================================================="

set -e
RANK_SIZE=$1
data_url=$2
pre_model=$3
train_url=$4

export RANK_SIZE=${RANK_SIZE}
export DEVICE_NUM=${RANK_SIZE}

rm -rf ./train_parallel
mkdir ./train_parallel
cp ./*.py ./train_parallel
cp -r ./src ./train_parallel
cp -r ./script ./train_parallel
cd ./train_parallel || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python3 -u train.py \
      --is_modelarts NO --distribution_flag YES --lr 0.0002 --decay_epoch 70 --epoch 80 \
      --data_url ${data_url} --pretrained_model ${pre_model} \
      --train_url ${train_url} --device_target GPU > output.log 2>&1 &
cd ..

