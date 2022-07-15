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
echo "bash scripts/run_distribute_train_ascend.sh [DEVICE_NUM] [CONFIG_PATH] [image_dir] [mindrecord_dir] [RANK_TABLE_FILE]"
echo "for example: bash scripts/run_distribute_train_ascend.sh 8 ./default_config.yaml ./MOT17DET/train/ ./MOT17DET/MindRecord_COCO_TRAIN /home/hccl_8p_01234567_192.168.88.13.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=600
export DEVICE_NUM=$1
export RANK_SIZE=$1
export RANK_TABLE_FILE=$5
CONFIG_PATH="$2"

OUTPUT_PATH="run_distribute_train"

rm -rf "$OUTPUT_PATH"
mkdir "$OUTPUT_PATH"
cp "$CONFIG_PATH" "$OUTPUT_PATH"

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH3=$(get_real_path $3)
PATH4=$(get_real_path $4)
echo $PATH3
echo $PATH4

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    # shellcheck disable=SC2115
    rm -rf $OUTPUT_PATH/$OUTPUT_PATH$i
    mkdir $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./*.py $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./*.yaml $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./*.ckpt $OUTPUT_PATH/$OUTPUT_PATH$i
    cp ./scripts/*.sh $OUTPUT_PATH/$OUTPUT_PATH$i
    cp -r ./src $OUTPUT_PATH/$OUTPUT_PATH$i
    cd $OUTPUT_PATH/$OUTPUT_PATH$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    touch log.txt
    python train.py  \
    --config_path="$CONFIG_PATH" \
    --image_dir="$PATH3" \
    --anno_path="$PATH3/shuffled_det_annotations.txt" \
    --mindrecord_dir="$PATH4" \
    --base_lr=0.06 \
    --save_checkpoint_path="./" \
    --run_distribute=True \
    --device_target="Ascend" \
    --device_num="$RANK_SIZE" > log.txt 2>&1 &
    cd ../../
done
