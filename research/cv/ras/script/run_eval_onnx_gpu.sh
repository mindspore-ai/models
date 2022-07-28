#!/usr/bin/env bash

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
    echo "===================================================================================================="
    echo "Please run the script as:"
    echo "bash script/run_eval_onnx_gpu.sh [data_url] [save_url] [onnx_file]"
    echo "for example: bash script/run_eval_onnx_gpu.sh /home/data/Test/ /home/data/results/ /home/data/models/RAS800.onnx"
    echo "**********
          data_url: The data_url directory is the directory where the dataset is located,and there must be two
                    folders, images and gts, under data_url;
          save_url: This is a save path of evaluation results;
          onnx_file: The save path of exported onnx model file.
**********"
    echo "===================================================================================================="
exit 1
fi

set -e
rm -rf output_eval_onnx
mkdir output_eval_onnx

data_url=$1
save_url=$2
onnx_file=$3

python3 -u eval_onnx.py --data_url ${data_url} --save_url ${save_url} --onnx_file ${onnx_file} --device_target GPU > output_eval_onnx/eval_onnx_log.log 2>&1 &

