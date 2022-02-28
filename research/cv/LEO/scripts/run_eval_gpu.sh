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
echo "============================================================================================================================"
echo "Please run the script as: "
echo "bash scripts/run_eval_gpu.sh [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [CKPT_FILE] "
echo "For example: bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1/xxx.ckpt "
echo "============ bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/1P_mini_5/xxx.ckpt "
echo "============ bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/1P_tiered_1/xxx.ckpt "
echo "============ bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/1P_tiered_5/xxx.ckpt "
echo "============================================================================================================================"
export  GLOG_v=3
export  DEVICE_TARGET=GPU
export  DATA_PATH=$1
export  DATA_NAME=$2
export  NUM_TR_EXAMPLES_PER_CLASS=$3
export  CKPT_FILE=$4
nohup python eval.py --device_target $DEVICE_TARGET \
                     --data_path $DATA_PATH \
                     --dataset_name $DATA_NAME \
                     --num_tr_examples_per_class $NUM_TR_EXAMPLES_PER_CLASS \
                     --ckpt_file $CKPT_FILE \
                     > ${DATA_NAME}_${NUM_TR_EXAMPLES_PER_CLASS}_eval.log 2>&1 &
