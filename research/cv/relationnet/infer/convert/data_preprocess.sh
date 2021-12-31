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
rm -rf ../data/input/data_preprocess_Result
rm -rf ../data/input/label
mkdir -p ../data/input/data_preprocess_Result
mkdir -p ../data/input/label
echo $HOME
python3 ../../preprocess.py --dataset_path=../data/input/dataset --data_output_path=../data/input/data_preprocess_Result --label_classses_output_path=./data/input/label
echo $HOME
