#!/bin/bash

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The number of parameters must be 5.
if [ $# -ne 5 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_PATH] [COARSE_PIPELINE_PATH] [FINE_PIPELINE_PATH] [COARSE_RESULT_PATH] [FINE_RESULT_PATH]"
  echo "Example: "
  echo "         bash build.sh ../input/data/nyu2_test ./config/CoarseNet.pipeline ./config/FineNet.pipeline ./coarse_infer_result ./fine_infer_result"

  exit 1
fi

# The path of a folder containing eval images.
image_path=$1
# The path of coarse_net_pipeline file.
coarse_pipeline_dir=$2
# The path of fine_net_pipeline file.
fine_pipeline_dir=$3
# The path of coarse_infer_result.
coarse_infer_result=$4
# The path of fine_infer_result.
fine_infer_result=$5

if [ ! -d $image_path ]
then
  echo "Please input the correct directory containing images."
  exit
fi

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)
echo "enter $CUR_PATH"

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}

# to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

if [ ! "${MX_SDK_HOME}" ]
then
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
fi

if [ ! "${MX_SDK_HOME}" ]
then
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
fi

python3 main_sdk.py --data_path=$image_path \
                  --Coarse_PL_PATH=$coarse_pipeline_dir \
                  --Fine_PL_PATH=$fine_pipeline_dir\
                  --coarse_infer_result_url=$coarse_infer_result\
                  --fine_infer_result_url=$fine_infer_result

exit 0