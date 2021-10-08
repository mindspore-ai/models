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
ROOT_PATH=$(pwd)
export DEVICE_ID=$1
if [ $# == 7 ]
then
    python3 ${ROOT_PATH}/$2 --data_path $3 --preset $4 \
--platform=Ascend --pretrain_ckpt $5 --is_numpy --output_path $7 >log_eval.log 2>&1 &
else
    python3 ${ROOT_PATH}/$2 --data_path $3 --preset $4 \
--platform=Ascend --pretrain_ckpt $5 --output_path $6 >log_eval.log 2>&1 &
fi