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

"""
download dataset file to ./data

"""

TRAIN_URL=https://dataset-bj.cdn.bcebos.com/duconv/train.txt.gz
DEV_URL=https://dataset-bj.cdn.bcebos.com/duconv/dev.txt.gz
TEST_1_URL=https://dataset-bj.cdn.bcebos.com/duconv/test_1.txt.gz

mkdir data
cd ./data
wget --no-check-certificate ${TRAIN_URL}
wget --no-check-certificate ${DEV_URL}
wget --no-check-certificate ${TEST_1_URL}

gunzip train.txt.gz
gunzip dev.txt.gz
gunzip test_1.txt.gz
mv ./test_1.txt ./test.txt
