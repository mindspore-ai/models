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
echo "Please run the script at the diractory same with train.py: "
echo "bash scripts/run_distribute_train_ascend.sh CONFIG_PATH DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50 RANK_TABLE_FILE RANK_SIZE"
echo "for example: bash scripts/run_distribute_train_ascend.sh ./configs/market1501_config.yml /path/to/dataset/ /path/to/output/ /path/to/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt rank_table_8pcs.json 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

config_path=$1
DATA_DIR=$2
OUTPUT_PATH=$3
PRETRAINED_RESNET50=$4
rank_table_8pcs_file=$5

EXEC_PATH=$(pwd)

export RANK_TABLE_FILE=${EXEC_PATH}/scripts/$rank_table_8pcs_file
export RANK_SIZE=$6

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ${EXEC_PATH}/train.py  \
      --config_path="$config_path" \
      --device_target="Ascend" \
      --data_dir="$DATA_DIR" \
      --ckpt_path="$OUTPUT_PATH" \
      --train_log_path="$OUTPUT_PATH" \
      --pre_trained_backbone="$PRETRAINED_RESNET50" \
      --lr_init=0.00140 \
      --lr_cri=1.0 \
      --ids_per_batch=8 \
      --is_distributed=1 > output.train.log 2>&1 &
    cd ../
done
