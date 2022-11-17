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
echo "bash run_distributed_train_moe.sh DATA_DIR RANK_TABLE_FILE DEVICE_NUM TYPE MODE STAGE_NUM MICRO_SIZE"
echo "PER_BATCH RANK_START LOCAL_DEVICE_NUM EXPERT_NUM ENABLE_ALLTOALL EXPERT_PARALLEL_NUM"
echo "for example:"
echo "#######no pipeline#######"
echo "#######run 60B model by 8 NPU#######"
echo "bash run_distributed_train_moe.sh /path/dataset /path/hccl.json 8 fp32 2.6B 1 1 1 0 8 36 0 1"
echo "It is better to use absolute path."
echo "Currently, pipeline parallel is not supported while running the shell."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
DATA_DIR=$1
export RANK_TABLE_FILE=$2
RANK_SIZE=$3
PARAM_INIT_TYPE=$4
MODE=$5
STAGE_NUM=$6
MICRO_SIZE=$7
PER_BATCH=$8
RANK_START=$9
LOCAL_DEVICE_NUM=${10}
EXPERT_NUM=${11}
ENABLE_ALLTOALL=${12}
EXPERT_PARALLEL_NUM=${13}

for((i=0;i<${LOCAL_DEVICE_NUM};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$[i+RANK_START]
    export DEVICE_ID=$i
    python ${ROOT_PATH}/train.py --distribute=true --device_num=$RANK_SIZE --data_url=$DATA_DIR --run_type=train \
    --param_init_type=$PARAM_INIT_TYPE --mode=$MODE --stage_num=$STAGE_NUM --micro_size=$MICRO_SIZE \
    --per_batch_size=$PER_BATCH --gradient_aggregation_group=2 \
    --opt_offload=0 --use_moe=1 --expert_num=$EXPERT_NUM \
    --enable_alltoall=$ENABLE_ALLTOALL --expert_parallel_num=$EXPERT_PARALLEL_NUM > log$i.log 2>&1 &
done
