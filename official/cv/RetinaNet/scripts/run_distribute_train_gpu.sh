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
echo "bash scripts/run_distribute_train_GPU.sh DEVICE_NUM MINDRECORD_DIR CONFIG_PATH VISIABLE_DEVICES PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: bash scripts/run_distribute_train_GPU.sh 8 /cache/mindrecord_dir/ /config/default_config_GPU.yaml 0,1,2,3,4,5,6,7 /opt/retinanet-500_458.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 4 ] && [ $# != 6 ]
then
    echo "Usage: bash scripts/run_distribute_train_GPU.sh [DEVICE_NUM] [MINDRECORD_DIR] \
    [CONFIG_PATH] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

export RANK_SIZE=$1
MINDRECORD_DIR=$2
CONFIG_PATH=$3
export CUDA_VISIBLE_DEVICES="$4"
PRE_TRAINED=$5
PRE_TRAINED_EPOCH_SIZE=$6

core_num=`cat /proc/cpuinfo |grep "processor"|wc -l`
process_cores=$(($core_num/$RANK_SIZE))

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

rm -rf LOG
mkdir ./LOG
cp ./*.py ./LOG
cp -r ./src ./LOG
cp -r ./scripts ./LOG
cp -r ./config ./LOG
cd ./LOG || exit

if [ $# == 4 ]
then
    mpirun -allow-run-as-root -n $1 --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --distribute=True  \
    --device_num=$RANK_SIZE  \
    --workers=$process_cores  \
    --config_path=$CONFIG_PATH \
    --mindrecord_dir=$MINDRECORD_DIR > log.txt 2>&1 &
fi

if [ $# == 6 ]
then
    mpirun -allow-run-as-root -n $1 --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --distribute=True  \
    --device_num=$RANK_SIZE  \
    --workers=$process_cores  \
    --config_path=$CONFIG_PATH \
    --mindrecord_dir=$MINDRECORD_DIR\
    --pre_trained=$PRE_TRAINED \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE > log.txt 2>&1 &
fi

cd ../
