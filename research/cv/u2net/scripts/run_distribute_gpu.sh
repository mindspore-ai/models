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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh CONTENT_PATH LABEL_PATH"
echo "for example: bash scripts/run_distribute_train_gpu.sh /path/to/content /path/to/label"
echo "=============================================================================================================="

if [ ! -d $1 ]
then
    echo "error: CONTENT_PATH=$1 is not a directory"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: LABEL_PATH=$2 is not a directory"
exit 1
fi
export RANK_SIZE=8
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
  python train.py  --run_distribute 1 --content_path $1 --label_path $2 --device_target GPU > output.log 2>&1 &
