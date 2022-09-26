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
import csv
import sys
import logging


logging.basicConfig(
    level=logging.INFO,
    filename='./sdk_infer_acc.log',
    filemode='w'
)

with open(sys.argv[1]) as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    sam_num, acc_num = 0, 0
    for row in f_csv:
        sam_num += 1
        print(sam_num)
        if row[0] == row[1]:
            acc_num += 1
accuracy = acc_num / sam_num
logging.info('eval result:{:.3f}%'.format(accuracy * 100))
