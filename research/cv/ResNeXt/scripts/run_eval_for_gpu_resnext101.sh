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
if [ $# != 4 ]
then
    echo "Usage: bash run_eval_for_gpu_resnext101.sh [DEVICE_ID] [EVAL_DATA_DIR] [CHECKPOINTPATH] [CONFIG_PATH]."
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

if [ ! -d "$(get_real_path $2)" ]
then
    echo "error: EVAL_DATA_DIR=$2 is not a directory"
    echo "Usage: bash run_eval_for_gpu_resnext101.sh [DEVICE_ID] [EVAL_DATA_DIR] [CHECKPOINTPATH] [CONFIG_PATH]."
exit 1
fi

if [ ! -f "$(get_real_path $3)" ]
then
    echo "error: CHECKPOINTPATH=$3 is not a file"
    echo "Usage: bash run_eval_for_gpu_resnext101.sh [DEVICE_ID] [EVAL_DATA_DIR] [CHECKPOINTPATH] [CONFIG_PATH]."
exit 1
fi

if [ ! -f "$(get_real_path $4)" ]
then
    echo "error: CONFIG_PATH=$4 is not a file"
    echo "Usage: bash run_eval_for_gpu_resnext101.sh [DEVICE_ID] [EVAL_DATA_DIR] [CHECKPOINTPATH] [CONFIG_PATH]."
exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
DATA_DIR=$(get_real_path $2)
PATH_CHECKPOINT=$(get_real_path $3)
CONFIG_PATH=$(get_real_path $4)
python eval.py  \
    --checkpoint_file_path=$PATH_CHECKPOINT \
    --data_path=$DATA_DIR \
    --config_path=$CONFIG_PATH > eval.log 2>&1 &

