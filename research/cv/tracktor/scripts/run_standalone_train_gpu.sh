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
if [ $# != 2 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]"
echo "for example: bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

DEVICE_ID="$1"
CONFIG_PATH="$2"

OUTPUT_PATH="run_standalone_train"

rm -rf "$OUTPUT_PATH"
mkdir "$OUTPUT_PATH"
cp "$CONFIG_PATH" "$OUTPUT_PATH"

export CUDA_VISIBLE_DEVICES="$DEVICE_ID"

python train.py \
  --device_target="GPU" \
  --save_checkpoint_path="$OUTPUT_PATH" \
  --config_path="$CONFIG_PATH" > "$OUTPUT_PATH"/log.txt 2>&1 &
