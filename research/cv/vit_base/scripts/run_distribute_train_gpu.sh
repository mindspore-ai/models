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
# ============================================================================
if [[ $# -ne 4 ]]; then
    echo "Usage: bash ./scripts/run_distribute_train_gpu.sh [DATASET_NAME] [DEVICE_NUM] [LR_INIT] [LOGS_CKPT_DIR]"
exit 1;
fi

export RANK_SIZE=$2

if [ !  -d "$4" ]; then
  mkdir "$4"
fi

mpirun -n $2 --allow-run-as-root\
    python train.py  \
    --device_target="GPU" \
    --dataset_name="$1" \
    --lr_init=$3 \
    --logs_dir=$4 \
    > ./"$4"/train.log 2>&1 &
