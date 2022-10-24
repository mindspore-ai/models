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

# The number of parameters must be 4.
if [ $# -ne 5 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_PATH] [PIPELINE_PATH] [OUTPUT_PATH/RESULT_PATH] [SAVE_PATH] [GT_PATH]"
  echo "Example: "
  echo "         bash run.sh ../data/BSR/BSDS500/data/images/test ../data/pipeline/hed.pipeline \
  ./result/hed_result ./result/hed_eval_result ../data/BSR/BSDS500/data/groundTruth/test"

  exit 1
fi

# The path of a folder containing eval images.
input_dir=$1
# The path of pipeline file.
pipeline_path=$2
# The path of a folder used to store all results.
output_dir=$3
# The path of a folder used to store all eval results.
save_dir=$4
# The path of a folder containing groundtruth.
gt_dir=$5

if [ ! -d $input_dir ]
then
  echo "Please input the correct directory containing images."
  exit
fi

if [ ! -d $output_dir ]
then
  mkdir -p $output_dir
fi

if [ ! -d $save_dir ]
then
  mkdir -p $save_dir
fi

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)
echo "enter $CUR_PATH"

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }


#to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

if [ ! "${MX_SDK_HOME}" ]
then
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
fi

if [ ! "${MX_SDK_HOME}" ]
then
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
fi

file_path="./sdk_eval.py"
target_dir="../../"
echo cp -r $file_path $target_dir
cp -r $file_path $target_dir

python3  main.py  --input_dir=$input_dir \
                  --pipeline_path=$pipeline_path \
                  --output_dir=$output_dir \

python3 ../../sdk_eval.py --result_dir=$output_dir \
                  --save_dir=$save_dir \
                  --gt_dir=$gt_dir

target_dir="../../sdk_eval.py"
echo rm -rf $target_dir
rm -rf $target_dir

exit 0