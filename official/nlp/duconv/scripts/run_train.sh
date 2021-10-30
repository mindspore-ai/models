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

"""
train model
"""
if [ $# -ne 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh run_train.sh [TASK_NAME] [DATA_PATH]"
    echo "for example: sh scripts/run_train.sh match_kn_gene"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi

if [ ! -f $2 ]
then
    echo "error: DATA_PATH=$2 is not a file"
exit 1
fi

TASK_NAME=$1
DATA_PATH=$2
OUTPUT_PATH=$3

cd ..
PWD_DIR=`pwd`
rm -rf $OUTPUT_PATH
mkdir $OUTPUT_PATH
save_path=$PWD_DIR/$OUTPUT_PATH
cd $save_path
rm -rf checkpoint
mkdir checkpoint
cd ..
python train.py --epoch=30 \
                --task_name=${TASK_NAME} \
                --max_seq_length=256 \
                --batch_size=128 \
                --train_data_file_path=$DATA_PATH \
                --save_checkpoint_path=$save_path/checkpoint \
                --config_path=default_config.yaml >$save_path/train.log 2>&1 &
