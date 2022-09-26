#!/bin/sh
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
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")

Info(){
    echo "Failed!"
    echo "Usage: bash run_eval.sh.sh [EPISODE_NUM] [DEVICE] [CKPT_PATH] [ENV_NAME]"
    echo "Example: bash run_eval.sh 50 GPU ../out/2s3z 2s3z"
}

if [ $# != 4 ];then
    Info
    exit 1
fi

export OMP_NUM_THREADS=10

python -s ${self_path}/../eval.py --episode_num=$1 --device=$2 --ckpt_path=$3 --env_name=$4> eval_log.txt 2>&1 &
