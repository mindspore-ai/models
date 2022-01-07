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

if [ $# != 7 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] [MODALITY] [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]"
  exit 1
fi

dataset_path=$1
dataset=$2
train_list_path=$3
train_list=$4
modality=$5
pretrained_path=$6
pretrained_path_name=$7

if [ ! -d $dataset_path ]; then
  echo "error: DATASET_PATH=$dataset_path is not a directory"
  exit 1
fi

if [ ! -d $pretrained_path ]; then
  echo "error: PRETRAINED_PATH=$pretrained_path is not a file"
  exit 1
fi

ulimit -u unlimited

cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 \
python train.py --data_url=$dataset_path \
   --platform=GPU \
   --run_distribute=True \
   --dataset=$dataset \
   --train_list_path=$train_list_path \
   --train_list=$train_list \
   --modality=$modality \
   --pretrained_path=$pretrained_path \
   --pre_trained_name=$pretrained_path_name > log_${modality}.txt 2>&1 &

echo "Training background..."