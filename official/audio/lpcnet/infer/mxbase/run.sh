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
LOOP=2620
data_path=$1
result_path=$2
for ((i=0; i<LOOP; i+=100))
do
  let j=i+100
  echo "begin is $i, end is $j"
  echo "if the program stop, you can restart from i by modify the start of loop!"
  ./build/Lpcnet ${data_path} ${result_path} $i $j
done
