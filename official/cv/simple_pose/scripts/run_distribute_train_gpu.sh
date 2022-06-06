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

if [ $# -lt 5 ]
then
    echo "Usage: \
          bash scripts/run_distributed_train_gpu.sh [CKPT_SAVE_DIR(relative)] [BATCH_SIZE] [END_EPOCH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]\
          "
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

cd $BASEPATH/../

if [ ! -d ./logs ];
then
    mkdir ./logs
fi

config_path="./default_config.yaml"
echo "config path is : ${config_path}"
ckpt_path="$1"
echo "ckpt_path is : ${ckpt_path}"
log_path="./logs/train_gpu_distributed.log"
echo "log_path is : ${log_path}"

echo "start gpu distributed training"
export CUDA_VISIBLE_DEVICES="$5"
export RANK_SIZE=$4
mpirun --allow-run-as-root -n $4 python train.py --device_target GPU --batch_size $2 --ckpt_save_dir ${ckpt_path} --run_distribute True --end_epoch $3 > ${log_path} 2>&1 &
