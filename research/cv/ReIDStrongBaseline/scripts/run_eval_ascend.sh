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

if [ $# != 3 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_eval_gpu.sh CONFIG_PATH CKPT_PATH DATA_DIR"
echo "for example: bash scripts/run_eval_gpu.sh ./configs/market1501_config.yml /your/path/checkpoint_file /path/to/dataset/"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
config_path=$(get_real_path "$1")

PATH1=$(get_real_path "$2")
echo "$PATH1"
DATA_DIR=$(get_real_path "$3")

python eval.py  \
    --config_path="$config_path" \
    --device_target="Ascend" \
    --data_dir="$DATA_DIR" \
    --eval_model="$PATH1" > output.eval.log 2>&1 &
