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

model_path=$1
output_model_name=$2

/usr/local/Ascend/atc/bin/atc \
--model=$model_path \
--framework=1 \
--output=$output_model_name \
--input_format=NCHW \
--input_shape="actual_input_1:1,3,112,112" \
--soc_version=Ascend310 \
--insert_op_conf=./aipp.config \
--output_type=FP32
