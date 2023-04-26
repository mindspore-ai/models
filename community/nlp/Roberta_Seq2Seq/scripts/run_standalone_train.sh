#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
if [ $# != 5 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train.sh  DEVICE_ID EPOCH_SIZE CONFIG_PATH DATA_PATH CHECKPOINT_PATH  "
echo "for example: bash run_standalone_train.sh 0 52  /path/config.yaml /path/ende-l128-mindrecord00 /path/checkpoint"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_standalone_train
mkdir run_standalone_train
cp -rf ./src/ train.py ./*.yaml ./run_standalone_train
cd run_standalone_train || exit

DEVICE_ID=$1
EPOCH_SIZE=$2
CONFIG_PATH=$3
DATA_PATH=$4
CHECKPOINT_PATH=$5
export DEVICE_ID=$DEVICE_ID
export RANK_ID=0
export RANK_SIZE=1
echo "start training for device $DEVICE_ID"
python train.py  \
        --config_path=$CONFIG_PATH \
        --epoch=$EPOCH_SIZE \
        --batch_size=16 \
        --device_id=$DEVICE_ID \
        --checkpoint_path=$CHECKPOINT_PATH \
        --label_smoothing=0.0 \
        --save_checkpoint_steps=50 \
        --keep_checkpoint_max=5 \
        --data_path=$DATA_PATH > log_train.txt 2>&1 &

cd ..
