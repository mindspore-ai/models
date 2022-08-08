#!/usr/bin/env bash
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

# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train.sh RANK_TABLE_FILE CFG DATA_DIR CHECKPOINT_FILE_PATH"
echo "for example: bash scripts/run_distribute_train.sh RANK_TABLE_FILE configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt"
echo "=============================================================================================================="
exit 1;
fi
if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi
ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=600
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$(realpath $1)
CFG=$(realpath $2)
DATA_DIR=$(realpath $3)
CHECKPOINT_FILE_PATH=$(realpath $4)
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    folder=train_parallel$i
    rm -rf ./$folder
    mkdir ./$folder
    cp -r configs ./$folder
    cp -r src ./$folder
    cp *.py ./$folder
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./$folder ||exit
    env > env.log
    python -u train.py --cfg "$CFG" \
        AVA.FRAME_DIR "${DATA_DIR}/frames" \
        AVA.FRAME_LIST_DIR "${DATA_DIR}/ava_annotations" \
        AVA.ANNOTATION_DIR "${DATA_DIR}/ava_annotations" \
        TRAIN.CHECKPOINT_FILE_PATH "$CHECKPOINT_FILE_PATH" > log_distributed_ascend 2>&1 &
    cd ..
done
