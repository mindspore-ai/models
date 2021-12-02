# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Beautify the results
"""

import sys
import json

try:
    result_path = sys.argv[1]
except IndexError:
    print("Please enter result file path.")
    exit(1)

with open(result_path, 'r') as f:
    d = json.load(f)
    print("\n================================")
    print(d['title'])
    print("--------------------------------")
    print(d['value'][0]['key'], ": ", d['value'][0]['value'])
    print(d['value'][1]['key'], ": ", d['value'][1]['value'])
    print("--------------------------------")
    for item in d['value'][2:]:
        print(item['key'], ": ", item['value'])
    print("================================\n")
