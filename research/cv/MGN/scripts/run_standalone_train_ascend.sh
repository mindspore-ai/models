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

if [ $# != 4 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50"
echo "bash scripts/run_standalone_train_ascend.sh 1 /data/dataset/Market-1501-v15.09.15/ output resnet50.ckpt "
echo "for example: bash scripts/run_distribute_train_ascend.sh 8 /path/to/market1501/ /path/to/output/ /path/to/pretrained_resnet50.pth HCCL_JSON"
echo "=============================================================================================================="
exit 1;
fi


get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

config_path=$(get_real_path "./configs/market1501_config.yml")

export RANK_SIZE=1
echo $RANK_SIZE
DATA_DIR=$(get_real_path "$2")
OUTPUT_PATH=$(get_real_path "$3")
PRE_TRAINED_PATH=$(get_real_path "$4")
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

export DEVICE_ID=$1
export RANK_ID=$1
rm -rf ./train
mkdir ./train
cd ./train || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python $BASE_PATH/../train.py  \
--config_path="$config_path" \
--data_dir="$DATA_DIR" \
--ckpt_path="$OUTPUT_PATH" \
--train_log_path="$OUTPUT_PATH" \
--pre_trained_backbone="$PRE_TRAINED_PATH" \
--device_target=Ascend \
--lr_init=0.00015 \
--ids_per_batch=12 \
--optimizer="adamw" \
--use_map=True \
--run_eval=True \
--decay_epochs="320,380" \
--max_epoch=400 \
--is_distributed=0 > train.log 2>&1 &
cd ..
