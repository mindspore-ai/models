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

if [ $# -lt 1 ]; then
    echo "Usage: bash run_standalone_eval_ascend.sh  [CKPT_FILE_PATH]
    CKPT_FILE_PATH is saved model ckpt file path."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

ckpt_path=$(get_real_path $1)

if [ ! -f ${ckpt_path} ]; then
    echo "Ckpt file does not exist."
exit 1
fi

cd ../
python eval.py --ckpt_path ${ckpt_path} > eval.log 2>&1 &
