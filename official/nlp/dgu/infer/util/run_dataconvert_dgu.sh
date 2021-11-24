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

set -e

if [ $# -ne 2 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_dataconvert_dgu.sh [TASK_TYPE] [MODE]"
    echo "for example: bash run_dataconvert_dgu.sh atis_intent test"
    echo "TASK_TYPE including [atis_intent, mrda, swda]"
    echo "MODE including [test, infer]"
    echo "=============================================================================================================="
exit 1
fi

TASK_TYPE=$1
MODE=$2

case $MODE in
  "test")
    OUTPUT_DIR="input"
    ;;
  "infer")
    OUTPUT_DIR="infer"
    ;;
  esac
# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

#to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python
cp ../../src/tokenizer.py ./

python3.7 data_processor_seq.py --task_name=${TASK_TYPE} --data_path=../data/rawdata/${TASK_TYPE} --vocab_file=../data/config/bert-base-uncased-vocab.txt --mode=${MODE} --max_seq_len=128 --output_path=../data/${OUTPUT_DIR}/${TASK_TYPE} 
exit 0
