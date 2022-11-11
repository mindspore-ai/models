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

echo "=============================================================================================================="
echo "Please run the script at the diractory same with train.py: "
echo "bash scripts/run_train_ascend.sh data_url pre_trained"
echo "For example: bash ./scripts/run_train_ascend.sh /path/dataset/ /path/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

data_url=$1
pre_trained=$2
rank_table_8pcs_file=$3
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=3600

EXEC_PATH=$(pwd)

export RANK_TABLE_FILE=${EXEC_PATH}/scripts/$rank_table_8pcs_file
export RANK_SIZE=8

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ${EXEC_PATH}/train.py --data_url=$data_url --pre_trained=$pre_trained > train$i.log 2>&1 &
    cd ../
done

rm -rf device0
mkdir device0
cp ./train.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python ${EXEC_PATH}/train.py --data_url=$data_url --pre_trained=$pre_trained > train0.log 2>&1
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
