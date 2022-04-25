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
"""Merge final annotation json"""
import os
import argparse
from glob import glob

from src.utils import txt_to_dict, dict_save_json, refine_json_again


def main(args):
    # merge all txt to one dict
    txt_root_path = args.txt_root_path
    base_json = args.base_json
    class_to_id = args.class_to_id
    dict_json_path = os.path.join(txt_root_path, 'all_test_txt.json')
    targe_save_path = os.path.join(txt_root_path, 'annotation_new.json')

    txt_list = glob(txt_root_path + "*.txt")
    return_dict = {}
    for each in txt_list:
        value = txt_to_dict(each)
        return_dict.update(value)

    sorted_dict = sorted(return_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    dict_save_json(sorted_dict, dict_json_path)

    # merge activate and base
    refine_json_again(class_to_id, base_json, dict_json_path, targe_save_path, rate=0.15, unlable_contain_all=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate final annotation')
    parser.add_argument('--class_to_id', default='./class_to_idx.json',
                        type=str, help='the class to id of imagenet dataset')
    parser.add_argument('--txt_root_path', type=str, help='txt root path of activate selecting')
    parser.add_argument('--base_json', type=str, help='annotation file generate from generage_anno.py')

    args_input = parser.parse_args()
    main(args_input)
