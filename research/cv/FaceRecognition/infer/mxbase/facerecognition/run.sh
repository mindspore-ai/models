#!/bin/bash

#coding = utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

bash build.sh
label_name=$1
model_path=$2
zj_list_path=$3
jk_list_path=$4
dis_list_path=$5

nohup ./build/resnet ${label_name} ${model_path} ${zj_list_path} ${jk_list_path} ${dis_list_path} &
