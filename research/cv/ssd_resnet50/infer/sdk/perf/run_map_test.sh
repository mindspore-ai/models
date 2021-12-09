#!/bin/bash

# Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
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

PY=/usr/bin/python3.7

export PYTHONPATH=${PYTHONPATH}:.

annotations_json=$1
det_result_json=$2
output_path_name=$3

${PY} generate_map_report.py \
--annotations_json=${annotations_json} \
--det_result_json=${det_result_json} \
--output_path_name=${output_path_name} \
--anno_type=bbox