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
'''
calculate AUC
'''

def calc_auc(raw_arr):
    '''
    calculate AUC
    :param raw_arr:
    :return:
    '''

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if abs(record[1] - 1.) < 0.000001:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if abs(record[1] - 1.) < 0.000001:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc
