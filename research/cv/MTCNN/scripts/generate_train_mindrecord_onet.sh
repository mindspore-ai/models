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
echo "bash generate_train_mindrecord_onet.sh PNET_CKPT RNET_CKPT"
echo "for example: bash generate_train_mindrecord_onet.sh pnet.ckpt rnet.ckpt"
echo "=============================================================================================================="

if [ $# -lt 2 ];
then
    echo "---------------------ERROR----------------------"
    echo "You must specify the checkpoint of PNet and RNet"
    exit
fi

PNET_CKPT=$1
RNET_CKPT=$2

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs

cd $PROJECT_DIR/../ || exit

python -m src.prepare_data.generate_ONet_data \
    --pnet_ckpt $PNET_CKPT \
    --rnet_ckpt $RNET_CKPT > $LOG_DIR/generate_onet_mindrecord.log 2>&1 &
echo "The data log is at /logs/generate_onet_mindrecord.log"
