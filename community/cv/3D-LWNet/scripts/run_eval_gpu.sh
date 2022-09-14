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
if [ $# -ne 3 ]; then
    echo "Usage: bash run_train_gpu.sh [CKPT_FILE] [DATASET] [DATA_PATH]
    DATASET must choose from ['Indian', 'Salinas', 'WHU']"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
data_path="$(get_real_path $3)"

if [ $# == 3 ]
then
    if [ $2 == "Indian" ]; then
        CONFIG_FILE="${BASE_PATH}/config/Indian_pines.yaml"
    elif [ $2 == "Salinas" ]; then
        CONFIG_FILE="${BASE_PATH}/config/Salinas.yaml"
    elif [ $2 == "WHU" ]; then
        CONFIG_FILE="${BASE_PATH}/config/WHU_Hi_HongHu.yaml"
    else
        echo "error: the selected dataset is not in supported set{Indian, Salinas, WHU}"
    exit 1
    fi
fi
python ${BASE_PATH}/../eval.py \
  --device_target="GPU" \
  --ckpt_file=$1 \
  --data_path=$data_path \
  --config_path=$CONFIG_FILE > eval.log 2>&1 &