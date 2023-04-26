#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
if [ $# != 6 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_eval.sh  DEVICE_ID DATA_PATH CKPT_PATH CONFIG_PATH VOCAB_FILE_PATH OUTPUT_FILE"
echo "for example: bash run_eval.sh 0 /your/path/data_path /your/path/checkpoint_file ./default_config.yaml ./vocab.json ./output_file"
echo "Note: set the checkpoint and dataset path in default_config.yaml"
echo "=============================================================================================================="
exit 1;
fi

DEVICE_ID=$1
DATA_PATH=$2
CHECKPOINT_PATH=$3
CONFIG_PATH=$4
VOCAB_PATH=$5
OUTPUT_PATH=$6
export DEVICE_ID=$DEVICE_ID
python eval.py  \
    --device_id=$DEVICE_ID \
    --config_path=$CONFIG_PATH \
    --data_path=$DATA_PATH \
    --model_file=$CHECKPOINT_PATH \
    --vocab_file_path=$VOCAB_PATH \
    --output_file=$OUTPUT_PATH > log_eval.txt 2>&1 &
