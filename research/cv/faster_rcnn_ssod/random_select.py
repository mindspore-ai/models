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
"""Random select train dataset"""
import argparse
import copy
import json
import random
import time

from pycocotools.coco import COCO


def random_select():
    """
    annotation json (coco format) -> all image ids ->
    random select ratio% label images with policy -> save new annotation json (coco format)

    inputs:
        ann_file(str): train annotation json file with coco format.
        label_ratio(int): set ratio of label images of all images.
        policy (int):
            policy=1 means ratio% labels and (100-ratio)% unlabels.
            policy=2 means ratio% labels and 100% unlabels.
        output_ann_file(str): new train annotation json file with coco format.
        seed (int): random seed
    """

    parser = argparse.ArgumentParser(description="random select ratio data for labels")
    parser.add_argument("--ann_file", type=str, required=True, help="ori annotation file.")
    parser.add_argument("--label_ratio", type=int, required=True, help="select ratio.")
    parser.add_argument("--policy", type=int, default=1,
                        help="select policy. policy=1 for ratio% labels and (100-ratio)% unlabels. "
                             "policy=2 for ratio% labels and 100% unlabels.")
    parser.add_argument("--output_ann_file", type=str, required=True, help="output annotation file.")
    parser.add_argument("--seed", type=int, default=10, help="random seed, -1 for time random.")
    args_opt = parser.parse_args()

    ann_file = args_opt.ann_file
    label_ratio = args_opt.label_ratio
    policy = args_opt.policy
    output_ann_file = args_opt.output_ann_file
    seed = args_opt.seed

    # random select
    if seed != -1:
        random.seed(seed)

    coco = COCO(ann_file)
    all_ids = coco.getImgIds()
    all_ids_num = len(all_ids)
    label_ids_num = int(label_ratio / 100.0 * all_ids_num)
    label_ids = random.sample(all_ids, label_ids_num)

    if policy == 1:
        generate_json_with_policy_1(ann_file, label_ids, output_ann_file)
    elif policy == 2:
        generate_json_with_policy_2(ann_file, label_ids, output_ann_file)


def generate_json_with_policy_1(ann_file, label_ids, output_ann_file):
    # generate new train annotation json with policy=1
    with open(ann_file, "r") as file_handle:
        ann_dict = json.load(file_handle)

    for anno in ann_dict["annotations"]:
        if anno["image_id"] not in label_ids:
            # label remain, unlabel do process
            anno['bbox'] = [0 for _ in range(4)]
            anno["category_id"] = -1
            anno["area"] = 0.0

    with open(output_ann_file, "w") as file_handle:
        json.dump(ann_dict, file_handle)


def generate_json_with_policy_2(ann_file, label_ids, output_ann_file):
    # generate new train annotation json with policy=2
    with open(ann_file, "r") as file_handle:
        ann_dict = json.load(file_handle)

    new_ann_list = []
    max_ann_id = 0
    for anno in ann_dict["annotations"]:
        if anno["image_id"] in label_ids:
            new_ann_list.append(anno)
        if max_ann_id < anno["id"]:
            max_ann_id = anno["id"]

    tmp_ann_dict = copy.deepcopy(ann_dict)
    for anno in tmp_ann_dict["annotations"]:
        anno['bbox'] = [0 for _ in range(4)]
        anno["category_id"] = -1
        anno["area"] = 0.0
        if anno["image_id"] in label_ids:
            max_ann_id += 1
            anno["id"] = max_ann_id
        new_ann_list.append(anno)

    ann_dict["annotations"] = new_ann_list

    with open(output_ann_file, "w") as file_handle:
        json.dump(ann_dict, file_handle)


if __name__ == '__main__':
    localtime_start = time.asctime(time.localtime(time.time()))
    print("[INFO] start time: {}".format(localtime_start))
    random_select()
    localtime_end = time.asctime(time.localtime(time.time()))
    print("[INFO] end time: {}".format(localtime_end))
