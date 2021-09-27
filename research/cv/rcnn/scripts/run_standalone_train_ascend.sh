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
if [ $# != 1 ]
then
    echo "Usage: sh run_standalone_train_ascend.sh [DEVICE_ID]"
exit 1
fi

export DEVICE_ID=$1
python ../train.py -d=${DEVICE_ID} --step=0 >train_log_finetune 2>&1 &
wait
python ../train.py -d=${DEVICE_ID} --step=1 >train_log_svm 2>&1 &
wait
python ../train.py -d=${DEVICE_ID} --step=2 >train_log_regression 2>&1 &