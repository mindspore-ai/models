#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
echo "bash run_distributed_train_ascend.sh DATA_PATH RANK_SIZE RANK_TABLE_FILE"
echo "For example: bash scripts/run_distributed_train_ascend.sh /path/dataset 8 /path/hccl.json"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

DATA_PATH=$1
RANK_SIZE=$2
RANK_TABLE_FILE=$3
export DATA_PATH=${DATA_PATH}
export RANK_SIZE=${RANK_SIZE}
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
export HCCL_CONNECT_TIMEOUT=600
ulimit -s 302400

EXEC_PATH=$(pwd)
CONFIG_PATH=${EXEC_PATH}/default_config.yaml

if [ ! -f "${RANK_TABLE_FILE}" ]
then
    echo "ERROR: ${RANK_TABLE_FILE} is not a valid path for RAND_TABKE_FILE, please check."
    exit 0
fi

if [ ! -d "${DATA_PATH}" ]
then
    echo "ERROR: ${DATA_PATH} is not a valid path for dataset, please check."
    exit 0
fi

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    export DEPLOY_MODE=0
    export GE_USE_STATIC_MEMORY=1
    echo "start training for device $i"
    mkdir -p ./ms_log
    export GLOG_log_dir=./ms_log
    export GLOG_logtostderr=0
    env > env$i.log
    python ../train.py --run_distribute True --config_path ${CONFIG_PATH} --platform Ascend  --dataset_path ${DATA_PATH} --rank_size ${RANK_SIZE} > train.log$i 2>&1 &
    cd ../
done
