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

if [ $# != 2 ]
then
    echo "Usage: bash run_standalone_train.sh [resume_path] [DEVICE_ID]."
    exit 1
fi
train_path="data/ucf101_rgb_train_split_1.txt"
val_path="data/ucf101_rgb_val_split_1.txt"
#--- training hyperparams ---
dataset_name="ucf101"
netType="ECO"
batch_size=16
learning_rate=0.001
num_segments=4
dropout=0.7
device_id=$2
resume=$1

python3 -u eval.py --dataset ${dataset_name} --modality RGB --train_list ${train_path} --val_list ${val_path} --arch ${netType} --num_segments ${num_segments}  --lr ${learning_rate} --num_saturate 5 --epochs 60 --batch-size ${batch_size} --dropout ${dropout}  --consensus_type identity --rgb_prefix img_  --no_partialbn True --nesterov True --evaluate True --resume ${resume} --device_id ${device_id} 1>log.txt 2>&1 &
