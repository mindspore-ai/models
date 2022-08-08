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

if [ $# != 5 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_ascend.sh DEVICE_NUM DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50 HCCL_JSON"
echo "bash scripts/run_distribute_train_ascend.sh 4 /data/dataset/Market-1501-v15.09.15/ output resnet50.ckpt /data/MGN_4/hccl_4p_0123_127.0.0.1.json"
echo "bash scripts/run_distribute_train_ascend.sh 8 /data/dataset/Market-1501-v15.09.15/ output resnet50.ckpt /data/MGN_COPY/rank_table_8pcs.json"
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

export RANK_SIZE=$1
echo $RANK_SIZE
DATA_DIR=$(get_real_path "$2")
OUTPUT_PATH=$(get_real_path "$3")
PRE_TRAINED_PATH=$(get_real_path "$4")
HCCL_JSON=$(get_real_path "$5")
export RANK_TABLE_FILE=$HCCL_JSON
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
rank_start=0

for((i=0; i<${RANK_SIZE}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python $BASE_PATH/../train.py  \
    --config_path="$config_path" \
    --data_dir="$DATA_DIR" \
    --ckpt_path="$OUTPUT_PATH" \
    --train_log_path="$OUTPUT_PATH" \
    --pre_trained_backbone="$PRE_TRAINED_PATH" \
    --device_target=Ascend \
    --lr_init=0.00025 \
    --ids_per_batch=6 \
    --optimizer="adamw" \
    --use_map=True \
    --run_eval=True \
    --decay_epochs="640,760" \
    --max_epoch=800 \
    --is_distributed=1 > train.log 2>&1 &
    cd ..
done
