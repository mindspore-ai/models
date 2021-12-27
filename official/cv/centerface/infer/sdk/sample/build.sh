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
# ===========================================================================

set -e

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

# compile
if g++ ../../mxbase/MxCenterfacePostProcessor.cpp ../../mxbase/MxImage.cpp ../../mxbase/MxUtil.cpp main.cpp -I ${MX_SDK_HOME}/include/ \
             -I ${MX_SDK_HOME}/opensource/include/ \
             -I ${MX_SDK_HOME}/opensource/include/opencv4 \
             -I ${ASCEND_HOME}/nnrt/latest/acllib/include \
             -L ${ASCEND_HOME}/nnrt/latest/acllib/lib64 \
             -L ${MX_SDK_HOME}/lib/ \
             -L ${MX_SDK_HOME}/opensource/lib/ \
             -L ${MX_SDK_HOME}/opensource/lib64/ \
             -std=c++11 \
             -D_GLIBCXX_USE_CXX11_ABI=0 \
             -Dgoogle=mindxsdk_private  \
             -fPIC -fstack-protector-all \
               -Wl,-z,relro,-z,now,-z,noexecstack -pie -g -Wall -lglog \
             -lmxbase -lstreammanager -lopencv_world -lcpprest -lgflags -lpthread -lmindxsdk_protobuf -lmxpidatatype -lascendcl\
             -o main;
then
  info "Build successfully."
else
  warn "Build failed."
fi
exit 0
