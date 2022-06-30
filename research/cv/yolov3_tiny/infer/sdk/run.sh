#!/bin/bash
# Copyright (c) 2022 Huawei Technologies Co., Ltd
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

dataset_path=$1
result_files=$2
pipeline_path=$3
ann_file=$4

python3 main.py --dataset_path ${dataset_path} \
                --result_files ${result_files} \
                --pipeline_path ${pipeline_path} \
                --ann_file ${ann_file}

exit 0