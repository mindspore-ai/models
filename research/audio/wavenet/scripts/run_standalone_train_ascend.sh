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
ROOT_PATH=$(pwd)
train_file=$1
DATA_DIR=$2
PRESET=$3
CKPT_DIR=$4
export DEVICE_ID=$5
export RANK_ID=0
export RANK_SIZE=1
python3 ${ROOT_PATH}/${train_file} --data_path $DATA_DIR --preset $PRESET \
--platform=Ascend --checkpoint_dir $CKPT_DIR >log_train.log 2>&1 &
