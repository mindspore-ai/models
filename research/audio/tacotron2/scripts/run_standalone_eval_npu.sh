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
root_path=$(pwd)
ckpt=$1
text=$2
out_dir=$3
fname=$4
device_id=$5
python3 ${root_path}/eval.py --ckpt_pth ${ckpt} --text "${text}" --out_dir ${root_path}/${out_dir} --fname ${fname} --device_id ${device_id}
