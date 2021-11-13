# coding = utf-8
"""
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
import sys

import numpy as np
import pandas as pd

age_lable = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
age_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gender_lable = ['Male', 'Female']
gender_count = [0, 0]
mask_lable = ['wearing mask', 'without mask']


def generatetxt(file, csvdata, data_dir):
    """ generate the label txt """
    for indata in csvdata:
        if indata[1] in age_lable:
            age = age_lable.index(indata[1])
            # record the number of every age groups
            age_count[age] += 1
        else:
            # ignore this label
            age = -1
            age_count[9] += 1
        if indata[2] in gender_lable:
            # record the count of gender
            gender = gender_lable.index(indata[2])
            gender_count[gender] += 1
        else:
            # ignore this label
            gender = -1
        strContent = data_dir + indata[0] + ' ' + str(age) + ' ' + str(gender) + ' ' + str(1)
        print(strContent)
        file.write(strContent)
        file.write('\n')
    print(gender_count)
    print(age_count)


if __name__ == '__main__':
    txt_dir = sys.argv[1]
    csv_path = sys.argv[2]
    data_Dir = sys.argv[3]
    # Read the label.txt file and create it if there is none.
    # 'a' means that the previous content will not be overwritten when it is written again
    f = open(txt_dir, 'a')
    data = np.array(pd.read_csv(csv_path))
    generatetxt(file=f, csvdata=data, data_dir=data_Dir)
