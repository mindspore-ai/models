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

cd ..

# help message
if [ $# != 2 ]; then
  echo "Usage: bash run_standalone_train.sh [device_id] [context]"
  exit 1
fi

ulimit -c unlimited
ulimit -n 65530
export SLOG_PRINT_TO_STDOUT=0
export  MS_DEV_ENABLE_FALLBACK=0

if [ ! -d "./output_standalone" ];then
  mkdir ./output_standalone
fi

nohup python ./train.py \
--device_id=$1 \
--context=$2\
--config_filename=data/model/dcrnn_la.yaml > ./output_standalone/dcrnn_device$1_log.txt 2>&1 &

echo 'start training...'



