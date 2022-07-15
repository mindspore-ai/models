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
echo "bash scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH]"
echo "for example: bash scripts/run_eval_ascend.sh 0 ./default_config.yaml"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

DEVICE_ID="$1"
CONFIG_PATH="$2"

OUTPUT_PATH="run_eval"

rm -rf "$OUTPUT_PATH"
mkdir "$OUTPUT_PATH"
cp "$CONFIG_PATH" "$OUTPUT_PATH"

export CUDA_VISIBLE_DEVICES="$DEVICE_ID"
export DEVICE_ID=$1

python eval.py \
  --config_path="$CONFIG_PATH" \
  --device_target="Ascend" \
  --device_id=$1 \
  --save_checkpoint_path="$OUTPUT_PATH" > "$OUTPUT_PATH"/log.txt 2>&1 &
