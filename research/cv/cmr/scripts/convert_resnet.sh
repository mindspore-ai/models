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
echo "bash convert_resnet.sh PTH_PATH CKPT_PATH"
echo "for example: bash convert_resnet.sh resnet50-19c8e357.pth pretrained_resnet50.ckpt"
echo "=============================================================================================================="

PTH_PATH=$1
CKPT_PATH=$2
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
DICT_FILE=$PROJECT_DIR/../data/resnet50_dict.json

if [ $# != 2 ]
then
    echo "Please specify the pth of PyTorch and ckpt of Mindspore"
    echo "Please try again"
    exit
fi

LOG_DIR=$PROJECT_DIR/../logs

python $PROJECT_DIR/../src/utils/pth2ckpt.py \
    --pth-path $PTH_PATH \
    --ckpt-path $CKPT_PATH \
    --dict-file $DICT_FILE > $LOG_DIR/convert_resnet.log 2>&1 &

echo "The convert_resnet.log file is at /logs/convert_resnet.log"
