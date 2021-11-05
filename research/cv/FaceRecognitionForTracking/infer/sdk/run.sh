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

image_path=../data/input/

set -e

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# compile plugin program
if [ -d ../util/plugins/build ] ;then
    rm -rf ../util/plugins/build;
fi
cmake -S ../util/plugins -B ../util/plugins/build
make -C ../util/plugins/build -j

if [ ! -d $MX_SDK_HOME/lib/plugins/ ] ;then
    echo "distance plugins/build dir not exists";
    exit 0;
fi
if [ -f ../util/plugins/build/libmxpi_transposeplugin_2.so ] ;then
    echo "copy file libmxpi_transposeplugins_2.so to plugins";
    cp ../util/plugins/build/libmxpi_transposeplugin_2.so $MX_SDK_HOME/lib/plugins/;
    echo "copy successfully";
fi

#to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

python3.7 main.py --dataset=$image_path
exit 0
