# coding=utf-8

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

import os

path = "./result"

filenames = os.listdir(path)
result = "./results.txt"

file = open(result, 'w+', encoding="utf-8")

for i in range(3003):
    filepath = path + '/'
    filepath = filepath + 'transformer_bs_1_'+str(i)+'.txt'
    originfile = open(filepath)
    for line in originfile.readlines():
        line = line.strip()
        file.write(line)
        file.write(' ')
    file.write('\n')

file.close()
