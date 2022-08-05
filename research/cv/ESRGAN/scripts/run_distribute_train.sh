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

if [ $# != 10 ]
then
    echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
export DISTRIBUTE=$2
export RANK_TABLE_FILE=$3
export LRPATH=$4
export GTPATH=$5
export VGGCKPT=$6
export VPSNRLRPATH=$7
export VPSNRGTPATH=$8
export VGANLRPATH=$9
export VGANGTPATH=${10}
CWD=`pwd`



for((i=0;i<RANK_SIZE;i++))
do
        export DEVICE_ID=$i
        rm -rf ./train_parallel$i
        mkdir ./train_parallel$i
        cp -r ../src ./train_parallel$i
        cp -r ../*.py ./train_parallel$i
        cd ./train_parallel$i || exit
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log
                python train.py --run_distribute=$DISTRIBUTE --device_num=$RANK_SIZE \
                                --device_id=$DEVICE_ID --train_LR_path=$LRPATH --train_GT_path=$GTPATH --vgg_ckpt=$VGGCKPT \
                                --val_PSNR_LR_path=$VPSNRLRPATH --val_PSNR_GT_path=$VPSNRGTPATH --val_GAN_LR_path=$VGANLRPATH \
                                --val_GAN_GT_path=$VGANGTPATH --best_ckpt_path=${CWD}/best_ckpt > train.log 2>&1 &
        cd ..
done
