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
    echo "Usage: bash run_eval.sh [CONFIG_PATH] [CKPT_PATH] [CUDA_VISIBLE_DEVICES]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

CONFIG_PATH=$(get_real_path $1)
CKPT_PATH=$(get_real_path $2)

export CUDA_VISIBLE_DEVICES=$3

if [ -d "eval" ];
then
    rm -rf ./eval
fi

mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cp -r ../config ./eval
cd ./eval || exit
env > env.log

echo "start inferring for device $CUDA_VISIBLE_DEVICES"
python eval.py --config_path=$CONFIG_PATH --ckpt_path=$CKPT_PATH --device_target='GPU' > log.txt 2>&1 &
