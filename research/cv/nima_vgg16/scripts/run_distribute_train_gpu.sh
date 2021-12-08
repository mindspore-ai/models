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
if [ $# != 1 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_gpu.sh DEVICE_NUM"
echo "for example: bash scripts/run_distribute_train_gpu.sh 8"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_distribute_train
mkdir run_distribute_train
cp -rf ./src/ train.py ./*.yaml ./run_distribute_train
cd run_distribute_train || exit

export RANK_SIZE=$1
echo $RANK_SIZE

mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
    python train.py  \
    --config_path="./config_dist_8_gpu.yaml" \
    --distribute="true" > log.txt 2>&1 &
