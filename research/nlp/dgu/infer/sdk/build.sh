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
    echo "bash build.sh [TASK_TYPE] [MODE]"
    echo "for example: bash build.sh atis_intent test"
    echo "TASK_TYPE including [atis_intent, mrda, swda]"
    echo "MODE including [test, infer]"
    echo "=============================================================================================================="
exit 1
fi

TASK_TYPE=$1
MODE=$2

case $TASK_TYPE in
  "atis_intent")
    LABEL_FILE="map_tag_intent_id.txt"
    ;;
  "mrda")
    LABEL_FILE="map_tag_mrda_id.txt"
    ;;
  "swda")
    LABEL_FILE="map_tag_swda_id.txt"
    ;;
  esac

case $MODE in
  "test")
    DATA_DIR="input"
    EVAL="true"
    ;;
  "infer")
    DATA_DIR="infer"
    EVAL="false"
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

python3.7 main.py --pipeline=../data/config/dgu_${TASK_TYPE}.pipeline --data_dir=../data/${DATA_DIR}/${TASK_TYPE} --label_file=../data/config/${LABEL_FILE} --output_file=./${TASK_TYPE}_output.txt --do_eval=${EVAL} --task_name=${TASK_TYPE}
exit 0
