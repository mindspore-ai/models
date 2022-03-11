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

wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gunzip reviews_Books_5.json.gz
gunzip meta_Books.json.gz
python ../src/process_data.py meta_Books.json reviews_Books_5.json
python ../src/local_aggretor.py
python ../src/split_by_user.py
python ../src/generate_voc.py
mkdir Books
mv ./*info ./Books
mv ./jointed* ./Books
mv ./local* ./Books
mv ./*.pkl ./Books