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
echo "bash run_standalone_train_ascend.sh DEVICE_TARGET DEVICE_ID EPOCH_SIZE GRADIENT_ACCUMULATE_STEP"
echo "for example: bash run_standalone_train_ascend.sh GPU 0 91 1"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_standalone_train
mkdir run_standalone_train
cp -rf ./src/ train.py ./*.yaml ./run_standalone_train
cd run_standalone_train || exit

export DEVICE_TARGET=$1
export DEVICE_ID=$2
EPOCH_SIZE=$3
export GRADIENT_ACCUMULATE_STEP=$4

export CUDA_VISIBLE_DEVICES="$2"

python train.py  \
    --config_path="./default_config.yaml" \
    --distribute="false" \
    --epoch_size=$EPOCH_SIZE \
    --device_target=$DEVICE_TARGET \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --checkpoint_path="" \
    --save_checkpoint_num=30 > log.txt 2>&1 &

cd ..