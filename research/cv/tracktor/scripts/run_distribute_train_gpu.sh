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
echo "bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [CONFIG_PATH] [LR]"
echo "for example: bash scripts/run_distributed_train_gpu.sh 8 ./default_config.yaml 0.008"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

export RANK_SIZE=$1
CONFIG_PATH="$2"
BASE_LR="$3"

OUTPUT_PATH="run_distribute_train"

rm -rf "$OUTPUT_PATH"
mkdir "$OUTPUT_PATH"
cp "$CONFIG_PATH" "$OUTPUT_PATH"

echo "start training on $RANK_SIZE devices"

mpirun -n $RANK_SIZE --output-filename "$OUTPUT_PATH"/log_output --allow-run-as-root --merge-stderr-to-stdout \
    python train.py  \
    --config_path="$CONFIG_PATH" \
    --save_checkpoint_path="$OUTPUT_PATH" \
    --run_distribute=True \
    --device_target="GPU" \
    --device_num="$RANK_SIZE" \
    --base_lr="$BASE_LR" > "$OUTPUT_PATH"/log.txt 2>&1 &
