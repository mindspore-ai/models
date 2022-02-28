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
echo "===================================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_train_gpu.sh [DEVICE_NUM] [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [SAVE_PATH] "
echo "For example: bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1"
echo " ============bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/1P_mini_5"
echo " ============bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/1P_tiered_1"
echo " ============bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/1P_tiered_5"  
echo "===================================================================================================================="
echo "Please run distributed training script as: "
echo "For example: bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/8P_mini_1 "
echo " ============bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/8P_mini_5"
echo " ============bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/8P_tiered_1"
echo " ============bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/8P_tiered_5"            
echo "===================================================================================================================="
export  DEVICE_NUM=$1
export  DEVICE_TARGET=GPU
export  DATA_PATH=$2
export  DATA_NAME=$3
export  NUM_TR_EXAMPLES_PER_CLASS=$4
export  SAVE_PATH=$5
export  GLOG_v=3
nohup mpirun --allow-run-as-root -n $DEVICE_NUM python train.py \
                                    --device_num $DEVICE_NUM \
                                    --device_target $DEVICE_TARGET \
                                    --data_path $DATA_PATH \
                                    --dataset_name $DATA_NAME \
                                    --num_tr_examples_per_class $NUM_TR_EXAMPLES_PER_CLASS \
                                    --save_path $SAVE_PATH \
                                    > ${DEVICE_NUM}P_${DATA_NAME}_${NUM_TR_EXAMPLES_PER_CLASS}_train.log 2>&1 &
