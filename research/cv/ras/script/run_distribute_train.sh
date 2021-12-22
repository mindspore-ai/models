#!/usr/bin/env bash

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



echo "===================================================================================================="
echo "Please run the script as:"
echo "bash script/run_distribute_train.sh [json_file] [RANK_SIZE] [data_url] [pre_model] [train_url]"
echo "for example: bash script/run_distribute_train.sh /home/rank8.json 8 /home/data/ /home/resent50.ckpt /home/data/models/"
echo " *****
      json_file: This is a json file for distributed training;
      RANK_SIZE: The parameter is numbers of device for distributed training;
      data_url: The data_url directory is the directory where the data set is located,and there must be two folders, images and labels, under data_url;
      pre_model: path of pretrained model;
      train_url: the save path of checkpoint file."
echo "===================================================================================================="

set -e
json_file=$1
RANK_SIZE=$2
data_url=$3
pre_model=$4
train_url=$5

export RANK_TABLE_FILE=${json_file}
export RANK_SIZE=${RANK_SIZE}


for((i=0;i<${RANK_SIZE};i++))
do
  rm -rf device$i
  mkdir device$i
  cp -r ./src ./device$i
  cp train.py ./device$i
  cd ./device$i
  export DEVICE_ID=$i
  export RANK_ID=$i
  python3 -u train.py --is_modelarts NO --distribution_flag YES --lr 0.0002 --decay_epoch [70] --epoch 80 --data_url ${data_url} --pretrained_model ${pre_model} --train_url ${train_url}  > output.log 2>&1 &
  cd ../
done

