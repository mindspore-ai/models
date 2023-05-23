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

export DEVICE_ID=$1
DATA_DIR=$2
PATH_CHECKPOINT=$3
PLATFORM=$4


python eval.py  \
    --checkpoint_file_path=$PATH_CHECKPOINT \
    --device_target=$PLATFORM \
    --data_path=$DATA_DIR \
    --device_target=$PLATFORM > eval.txt 2>&1 &
