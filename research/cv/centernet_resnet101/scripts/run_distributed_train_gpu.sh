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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_train_gpu.sh MINDRECORD_DIR DEVICE_NUM LOAD_CHECKPOINT_PATH"
echo "for example: bash run_distributed_train_gpu.sh /path/mindrecord_dataset 8 /path/load_ckpt"
echo "if no ckpt, just run: bash run_distributed_train_gpu.sh /path/mindrecord_dataset 8"
echo "=============================================================================================================="

MINDRECORD_DIR=$1
RANK_SIZE=$2
if [ $# == 3 ];
then
    LOAD_CHECKPOINT_PATH=$3
else
    LOAD_CHECKPOINT_PATH=""
fi

mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
LOG_DIR=$PROJECT_DIR/../logs
if [ ! -d $LOG_DIR ]
then
    mkdir $LOG_DIR
fi
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export RANK_SIZE=$RANK_SIZE

mpirun -n $RANK_SIZE --allow-run-as-root python ${PROJECT_DIR}/../train.py  \
    --distribute=true \
    --device_num=$RANK_SIZE \
    --device_target=GPU \
    --need_profiler=false \
    --profiler_path=./profiler \
    --enable_save_ckpt=true \
    --do_shuffle=true \
    --enable_data_sink=false \
    --data_sink_steps=-1 \
    --epoch_size=330 \
    --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
    --save_checkpoint_steps=3664 \
    --save_checkpoint_num=5 \
    --mindrecord_dir=$MINDRECORD_DIR \
    --mindrecord_prefix="coco_det.train.mind" \
    --visual_image=false \
    --save_result_dir="" >${LOG_DIR}/distributed_training_gpu_log.txt 2>&1 &
