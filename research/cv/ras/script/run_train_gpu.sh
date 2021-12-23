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
echo "bash script/run_train_gpu.sh [device_id] [lr] [data_url] [pre_model] [train_url]"
echo "for example: bash script/run_train_gpu.sh 5 0.00005 /home/data/ /home/resent50.ckpt /home/data/models/"
echo " *****
      device_id: The parameter is device's ID for training;
      lr: learning_rate
      data_url: The data_url directory is the directory where the data set is located,and there must be two folders, images and labels, under data_url;
      pre_model: path of pretrained model;
      train_url: the save path of checkpoint file."
echo "===================================================================================================="


set -e
rm -rf output
mkdir output

device_id=$1
lr=$2
data_url=$3
pre_model=$4
train_url=$5

python3 -u train.py --is_modelarts NO --distribution_flag NO --device_id ${device_id} --lr ${lr} --data_url ${data_url} --pretrained_model ${pre_model} --train_url ${train_url} --device_target GPU> output/output.log 2>&1 &
