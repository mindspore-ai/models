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
    echo "Usage: bash run_eval_onnx.sh [DATA_ROOT] [DATA_LST] [FILE_NAME]"
exit 1
fi


EXECUTE_PATH=$(pwd)
eval_path=${EXECUTE_PATH}/onnx_eval

if [ -d ${eval_path} ]; then
  rm -rf ${eval_path}
fi
mkdir -p ${eval_path}

python ${EXECUTE_PATH}/../eval_onnx.py --data_root=$1  \
                    --data_lst=$2  \
                    --batch_size=8  \
                    --device_target GPU \
                    --file_name=$3 >${eval_path}/eval_log 2>&1 &

