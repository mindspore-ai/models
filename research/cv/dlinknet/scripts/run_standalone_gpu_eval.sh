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

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 5 ] && [ $# != 6 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_gpu_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH] [DEVICE_ID](option, default is 0)"
    echo "for example: bash run_standalone_gpu_eval.sh /path/to/data/ /path/to/label/ /path/to/checkpoint/ /path/to/predict/ /path/to/config/ 0"
    echo "=============================================================================================================="
    exit 1
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export DEVICE_ID=0
if [ $# != 5 ]
then
  export DEVICE_ID=$6
fi
rm -rf "$4"
mkdir "$4"
DATASET=$(get_real_path $1)
LABEL_PATH=$(get_real_path $2)
CHECKPOINT=$(get_real_path $3)
PREDICT_PATH=$(get_real_path $4)
CONFIG_PATH=$(get_real_path $5)
echo "========== start run evaluation ==========="
echo "please get log at eval.log"
python ${PROJECT_DIR}/../eval.py --data_path=$DATASET --label_path=$LABEL_PATH --trained_ckpt=$CHECKPOINT --predict_path=$PREDICT_PATH --config_path=$CONFIG_PATH --device_target=GPU > eval.log 2>&1 &
