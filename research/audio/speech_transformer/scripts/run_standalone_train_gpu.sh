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
if [ $# != 3 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [GRADIENT_ACCUMULATE_STEP]"
echo "for example: bash run_standalone_train_gpu.sh 0 150 1"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_standalone_train
mkdir run_standalone_train
cp -rf ./src/ train.py ./*.yaml ./run_standalone_train
cd run_standalone_train || exit

export DEVICE_ID=$1
EPOCH_SIZE=$2
export GRADIENT_ACCUMULATE_STEP=$3

export CUDA_VISIBLE_DEVICES="$1"

python train.py  \
    --config_path="./default_config.yaml" \
    --distribute="false" \
    --epoch_size=$EPOCH_SIZE \
    --device_target="GPU" \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --checkpoint_path="" \
    --ckpt_interval=1 \
    --save_checkpoint_num=10 > log.txt 2>&1 &

cd ..
