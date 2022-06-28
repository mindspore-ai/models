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
if [ $# != 3 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_gpu.sh DEVICE_ID EPOCH DATA_PATH"
echo "for example: bash run_standalone_train_gpu.sh 1 20000 '/your/path/omniglot/'"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_standalone_train
mkdir run_standalone_train
cp -rf ../src ./run_standalone_train
cp -rf ../omniglot_train.py ./run_standalone_train
cd run_standalone_train || exit

export DEVICE_ID=$1
export EPOCH_SIZE=$2
export DATA_PATH=$3

python omniglot_train.py  \
    --device_id=$DEVICE_ID \
    --device_target="GPU" \
    --epoch=$EPOCH_SIZE \
    --data_path=$DATA_PATH > log.txt 2>&1 &   
cd ..
