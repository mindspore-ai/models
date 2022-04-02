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

if [ $# != 1 ]; then 
    echo "Usage: bash run_standalone_train_gpu.sh [DATA_PATH]"
exit 1
fi

DATA_PATH=$1
echo $DATA_PATH

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a directory."
exit 1
fi

export DEVICE_NUM=1
export RANK_SIZE=1
export DEVICE_ID=0
export RANK_ID=0

rm -rf ../src/deep/train_standalone
mkdir ../src/deep/train_standalone
cp ../src/deep/*.py ../src/deep/train_standalone
cp ../src/deep/*.yaml ../src/deep/train_standalone
cd ../src/deep/train_standalone || exit

echo "start standalone training with $DEVICE_NUM GPUs."

python train.py --data_url=${DATA_PATH} \
                --train_url="" \
                --device "GPU" \
                --run_distribute=False > out.log 2>&1 &