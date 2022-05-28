#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

echo "- Downloading enwik8 (Character)"
if [[ ! -d 'enwik8' ]]; then
    mkdir -p enwik8
    cd enwik8
    wget --continue http://mattmahoney.net/dc/enwik8.zip --no-check-certificate
    wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py --no-check-certificate
    python3 prep_enwik8.py
    cd ..
fi

echo "- Downloading text8 (Character)"
if [[ ! -d 'text8' ]]; then
    mkdir -p text8
    cd text8
    wget --continue http://mattmahoney.net/dc/text8.zip --no-check-certificate
    python ../../prep_text8.py
    cd ..
fi

echo "---"
echo "Happy language modeling :)"
