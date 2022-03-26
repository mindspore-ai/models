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
echo "bash script/run_eval_gpu.sh [device_id] [data_url] [train_url] [model_path] [pre_model]"
echo "for example: bash script/eval_gpu.sh 5 /home/data/Test/ /home/data/results/ /home/data/models/RAS800.ckpt /home/data/resnet50.ckpt"
echo " *****
      device_id: The device id for evaluation;
      data_url: The data_url directory is the directory where the dataset is located,and there must be two folders, images and gts, under data_url;
      train_url: This is a save path of evaluation results;
      model_path: the save path of checkpoint file produced by the RAS during training process;
      pre_model: path of pretrained model.
      "
echo "===================================================================================================="

set -e
rm -rf output_eval
mkdir output_eval

device_id=$1
data_url=$2
train_url=$3
model_path=$4
pre_model=$5

python3 -u eval.py --device_id ${device_id} --data_url ${data_url} --train_url ${train_url} --model_path ${model_path} --pre_model ${pre_model} --device_target GPU > output_eval/eval_output.log 2>&1 &

