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

if [ $# != 3 ]
then
    echo "Usage: bash scripts/run_standalone_train.sh [BERT_CKPT] [DEVICE_ID] [PYNATIVE]"
exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"

rm -rf $DIR/output_standalone
mkdir $DIR/output_standalone

nohup python train.py --bert_ckpt $1 --device_id $2 --pynative $3 >>$DIR/output_standalone/device_log.txt 2>&1 &
