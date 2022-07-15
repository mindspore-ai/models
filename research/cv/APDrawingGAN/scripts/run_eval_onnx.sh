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


if [ $# != 5 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_eval_onnx.sh [DATA_PATH] [LM_PATH] [BG_PATH] [RESULT_PATH] [ONNX_PATH]"
    echo "for example: bash scripts/run_eval_onnx.sh test_dataset/data/test_single/ test_dataset/landmark/ALL test_dataset/mask/ALL test_result onnx_file/"
    echo "=============================================================================================================="
exit 1
fi

DATA_PATH=$1
LM_PATH=$2
BG_PATH=$3
RESULT_PATH=$4
ONNX_PATH=$5


get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

python eval_onnx.py  \
  --dataroot $DATA_PATH \
  --lm_dir $LM_PATH\
  --bg_dir $BG_PATH\
  --results_dir $RESULT_PATH\
  --onnx_path $ONNX_PATH \
  > eval_onnx.log 2>&1 &
