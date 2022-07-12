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

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: bash run_ckpt_convert.sh [PYTORCH_FILE_PATH] [MINDSPORE_FILE_PATH]
    PYTORCH_FILE_PATH is pytorch pretrained model ckpt file path.
    MINDSPORE_FILE_PATH is mindspore pretrained model ckpt file path."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

torch_file_path=$(get_real_path $1)

if [ ! -f ${torch_file_path} ]; then
    echo "Pytorch pretrained model ckpt file path does not exist."
exit 1
fi

mindspore_file_path=$(get_real_path $2)

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

python3 ${BASEPATH}/../src/model_utils/ckpt_convert.py ${torch_file_path} ${mindspore_file_path}
