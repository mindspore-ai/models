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
echo "bash run.sh [DEVICE_NUM] [MINDRECORD_PATH] [DATA_PATH] [DATA_TYPE]"
echo "For example: bash scripts/run_distribute_train_books_gpu.sh 8 ./dataset_mindrecord ./Books Books"
echo "It is better to use the absolute path.After running the script, the network runs in the background,the log will be generated in ms_log/log_dien_distribute.log"
echo "=============================================================================================================="
export DEVICE_NUM=$1
MINDRECORD_PATH=$2
DATA_PATH=$3
DATA_TYPE=$4
EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
python -u train.py \
    --device_target=GPU \
    --mindrecord_path=$MINDRECORD_PATH \
    --dataset_type=$DATA_TYPE \
    --dataset_file_path=$DATA_PATH \
    --epoch_size=4 \
    --base_lr=0.003 \
    --run_distribute=True > ms_log/log_dien_distribute.log 2>&1 &