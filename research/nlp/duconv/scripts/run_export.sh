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
export model

"""
if [ $# != 3 ]; then
  echo "Usage: sh run_distribute_train.sh [TASK] [CHECK_POINT_PATH] [FILE_FORMAT]"
  exit 1
fi
TASK_NAME=$1
CHECK_POINT_PATH=$2
FILE_FORMAT=$3
python export.py --task_name=${TASK_NAME} \
                 --max_seq_length=256 \
                 --batch_size=1 \
                 --checkpoint_file=$CHECK_POINT_PATH \
                 --device_id=0\
                 --file_format=$FILE_FORMAT > export_model.log 2>&1 &
