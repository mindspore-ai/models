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
"""Select top valuable data for second stage train"""
import argparse
import copy
import json
import operator
import time

from pycocotools.coco import COCO


def pick_select():
    """
    1. coco(train annotation json) -> top value file -> pick select b% -> outputs
    2. coco(train annotation json) -> combine_ann_file with a% labels and top value file
                                   -> pick select b% (include a%)-> outputs

    inputs:
        train annotation json with a% labels and (100-a)% unlabels (file)
        pick ratio b% labels (float)
        top value file (file)
        policy (int)
            policy=1 for ratio% labels and (100-ratio)% unlabels.
            policy=2 for ratio% labels and 100% unlabels.

    outputs:
        new train annotation json with (a+b)% labels (file), related to new train dir including label and unlabel
    """

    parser = argparse.ArgumentParser(description="random select ratio data for labels")
    parser.add_argument("--ann_file", type=str, required=True, help="ori annotation file.")
    parser.add_argument("--combine_ann_file", type=str, default="", help="labels and unlabels annotation file.")
    parser.add_argument("--pick_ratio", type=int, required=True, help="select ratio.")
    parser.add_argument("--top_value_file", type=str, required=True, help="top value file.")
    parser.add_argument("--policy", type=int, default=1,
                        help="select policy. policy=1 for ratio% labels and (100-ratio)% unlabels. "
                             "policy=2 for ratio% labels and 100% unlabels.")
    parser.add_argument("--output_ann_file", type=str, required=True, help="output annotation file.")
    args_opt = parser.parse_args()

    ann_file = args_opt.ann_file
    combine_ann_file = args_opt.combine_ann_file
    pick_ratio = args_opt.pick_ratio
    top_value_file = args_opt.top_value_file
    policy = args_opt.policy
    output_ann_file = args_opt.output_ann_file

    with open(top_value_file, "r") as file_handle:
        top_value_dict = json.load(file_handle)

    top_value_dict_sorted = sorted(top_value_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_value_files = [x[0] for x in top_value_dict_sorted]

    coco = COCO(ann_file)
    all_ids = coco.getImgIds()
    all_ids_num = len(all_ids)
    pick_label_ids_num = int(pick_ratio / 100.0 * all_ids_num)

    label_ids = []
    if combine_ann_file:
        # already has a% labels
        combine_coco = COCO(combine_ann_file)
        combine_all_ids = combine_coco.getImgIds()
        combine_label_ids, _ = divide_label_unlabel(combine_coco, combine_all_ids)
        if pick_label_ids_num > len(combine_all_ids):
            raise Exception("pick_label_ids_num {} exceeds combine_all_ids_num {}."
                            .format(pick_label_ids_num, len(combine_all_ids)))
        if pick_label_ids_num <= len(combine_label_ids):
            raise Exception("pick_label_ids_num {} less than combine_label_ids_num {}."
                            .format(pick_label_ids_num, len(combine_label_ids)))

        # pick b% labels, not includes a% labels
        for top_value_img_name in top_value_files:
            for img_id in all_ids:
                if img_id in combine_label_ids:
                    continue
                img_name = coco.loadImgs(img_id)[0]["file_name"]
                if img_name == top_value_img_name:
                    label_ids.append(img_id)
                    break
            if len(label_ids) == pick_label_ids_num - len(combine_label_ids):
                break

        if len(label_ids) != pick_label_ids_num - len(combine_label_ids):
            raise Exception("picked num {} not equal expected pick num {}."
                            .format(len(label_ids), pick_label_ids_num - len(combine_label_ids)))

        # sum a% and b%
        label_ids.extend(combine_label_ids)
        label_ids = list(set(label_ids))
        if len(label_ids) != pick_label_ids_num:
            raise Exception("picked sum num {} not equal expected sum num {}."
                            .format(len(label_ids), pick_label_ids_num))

    else:
        # pick b% labels
        for top_value_img_name in top_value_files:
            for img_id in all_ids:
                img_name = coco.loadImgs(img_id)[0]["file_name"]
                if img_name == top_value_img_name:
                    label_ids.append(img_id)
                    break
            if len(label_ids) == pick_label_ids_num:
                break

        if len(label_ids) != pick_label_ids_num:
            raise Exception("picked num {} not equal expected pick num {}.".format(len(label_ids), pick_label_ids_num))

    print("label_ids len: {}".format(len(label_ids)))
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


def divide_label_unlabel(coco, dataset_ids):
    # process new type annotation file which includes label and unlabel info
    ann_ids = coco.getAnnIds(imgIds=dataset_ids)
    annotations = coco.loadAnns(ann_ids)

    unlabels_cmp_category_id = -1
    unlabels_cmp_bbox = [0 for _ in range(4)]
    labels = []
    unlabels = []
    for anno in annotations:
        if anno["category_id"] == unlabels_cmp_category_id and operator.eq(anno["bbox"], unlabels_cmp_bbox):
            unlabels.append(anno["image_id"])
        else:
            labels.append(anno["image_id"])

    labels = list(set(labels))
    unlabels = list(set(unlabels))
    return labels, unlabels


if __name__ == '__main__':
    localtime_start = time.asctime(time.localtime(time.time()))
    print("[INFO] start time: {}".format(localtime_start))
    pick_select()
    localtime_end = time.asctime(time.localtime(time.time()))
    print("[INFO] end time: {}".format(localtime_end))
