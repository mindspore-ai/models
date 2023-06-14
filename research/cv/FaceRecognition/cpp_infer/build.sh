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
# ============================================================================
if [ -d out ]; then
    rm -rf out
fi

mkdir out
cd out || exit

if [ -f "Makefile" ]; then
  make clean
fi

if [ $MS_LITE_HOME ];then
  MINDSPORE_PATH=$MS_LITE_HOME/runtime
else
  MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
  if [[ ! $MINDSPORE_PATH ]];then
      MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
  fi
fi
cmake .. -DMINDSPORE_PATH=$MINDSPORE_PATH
make
