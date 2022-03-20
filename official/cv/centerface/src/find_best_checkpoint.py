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
"""find best checkpoint"""

import os
import argparse

def find_ckpt(file_path, arg):
    with open(file_path) as f:
        str_result_list = f.readlines()

    easy_list, medium_list, hard_list, path_list = [], [], [], []
    easy_index, medium_index, hard_index, sum_index = -1, -1, -1, -1
    easy_max, medium_max, hard_max, sum_max = -1, -1, -1, -1
    index = -1
    for i in range(len(str_result_list)):
        if str_result_list[i].startswith("==================== Results"):
            index += 1
            path_list.append(str_result_list[i][:-1])
            easy_ap = float(str_result_list[i + 1][15:-1])
            medium_ap = float(str_result_list[i + 2][15:-1])
            hard_ap = float(str_result_list[i + 3][15:-1])
            sum_ap = easy_ap + medium_ap + hard_ap
            easy_list.append(easy_ap)
            medium_list.append(medium_ap)
            hard_list.append(hard_ap)
            if easy_ap >= arg.filter_easy \
                    and medium_ap >= arg.filter_medium \
                    and hard_ap >= arg.filter_hard \
                    and sum_ap >= arg.filter_sum:
                if easy_ap >= easy_max:
                    easy_max = easy_ap
                    easy_index = index
                if medium_ap >= medium_max:
                    medium_max = medium_ap
                    medium_index = index
                if hard_ap >= hard_max:
                    hard_max = hard_ap
                    hard_index = index
                if sum_ap >= sum_max:
                    sum_max = sum_ap
                    sum_index = index

    if easy_index == -1:
        print("\nCannot find a checkpoint that meets the filter requirements.")
    else:
        print("\nThe best easy result:", flush=True)
        print(path_list[easy_index], flush=True)
        print("Easy   Val AP: {}".format(easy_list[easy_index]), flush=True)
        print("Medium Val AP: {}".format(medium_list[easy_index]), flush=True)
        print("Hard   Val AP: {}".format(hard_list[easy_index]), flush=True)
        print("=================================================", flush=True)

        print("\nThe best medium result:", flush=True)
        print(path_list[medium_index], flush=True)
        print("Easy   Val AP: {}".format(easy_list[medium_index]), flush=True)
        print("Medium Val AP: {}".format(medium_list[medium_index]), flush=True)
        print("Hard   Val AP: {}".format(hard_list[medium_index]), flush=True)
        print("=================================================", flush=True)

        print("\nThe best hard result:", flush=True)
        print(path_list[hard_index], flush=True)
        print("Easy   Val AP: {}".format(easy_list[hard_index]), flush=True)
        print("Medium Val AP: {}".format(medium_list[hard_index]), flush=True)
        print("Hard   Val AP: {}".format(hard_list[hard_index]), flush=True)
        print("=================================================", flush=True)

        print("\nThe best sum result:", flush=True)
        print(path_list[sum_index], flush=True)
        print("Easy   Val AP: {}".format(easy_list[sum_index]), flush=True)
        print("Medium Val AP: {}".format(medium_list[sum_index]), flush=True)
        print("Hard   Val AP: {}".format(hard_list[sum_index]), flush=True)
        print("=================================================", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file_path', default='', help='result file')
    parser.add_argument('--filter_easy', default=0.0, type=float, help='filter easy')
    parser.add_argument('--filter_medium', default=0.0, type=float, help='filter medium')
    parser.add_argument('--filter_hard', default=0.0, type=float, help='filter hard')
    parser.add_argument('--filter_sum', default=0.0, type=float, help='filter sum')
    args = parser.parse_args()

    if os.path.isfile(args.result_file_path):
        find_ckpt(args.result_file_path, args)
    else:
        raise FileNotFoundError("{} not found.".format(args.result_file_path))
