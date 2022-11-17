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

if [ $# -ne 2 ]
then 
    echo "Usage: bash scripts/run_distribute_gpu.sh [DATA_PATH] [CKPT_PATH]"
exit 1
fi
DATA_PATH=$1
CKPT_PATH=$2


export RANK_SIZE=8

echo "======start training======"

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python ./train.py \
  --data_url=$DATA_PATH \
  --device_target="GPU" \
  --train_url=$CKPT_PATH \
  --nepoch=50 > log_distribution_gpu 2>&1 &
cd ..