# Copyright 2021 Huawei Technologies Co., Ltd
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
""" generate the interactive segmentation format of augmented pascal dataset """
import os
import shutil
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat


def save_image(path, img, if_pal=False):
    """save image with palette"""
    img = Image.fromarray(img)
    if if_pal:
        img.putpalette([0, 0, 0, 128, 0, 0] + [0, 0, 0] * 253 + [224, 224, 192])
    img.save(path)


def get_list_from_file(file):
    """get list from txt file"""
    with open(file) as f:
        lines = f.read().splitlines()
    return lines


def process(img_ids, split):
    """ process dataset split"""
    global pascal_all_ids, pascal_path, sbd_path, dataset_path
    f_output = open(dataset_path / "list" / (split + ".txt"), "w")
    for img_id in tqdm(img_ids):
        if img_id in pascal_all_ids:
            img_path = pascal_path / "JPEGImages" / (img_id + ".jpg")
            gt_path = pascal_path / "SegmentationObject" / (img_id + ".png")
            gt = np.array(Image.open(gt_path))
        else:
            img_path = sbd_path / "img" / (img_id + ".jpg")
            gt_path = sbd_path / "inst" / (img_id + ".mat")
            gt = loadmat(gt_path)["GTinst"][0]["Segmentation"][0]

        shutil.copyfile(img_path, dataset_path / "img" / (img_id + ".jpg"))

        for i in set(gt.flat):
            if i == 0 or i > 254:
                continue
            id_ins = img_id + "#" + str(i).zfill(3)
            f_output.write(id_ins + "\n")
            gt_ins = (gt == i).astype(np.uint8)
            gt_ins[gt == 255] = 255
            save_image(dataset_path / "gt" / (id_ins + ".png"), gt_ins, if_pal=True)

    f_output.close()


if __name__ == "__main__":
    # parameters of datasets path
    parser = argparse.ArgumentParser(description="Generate ISF Dataset PASCAL_SBD")
    parser.add_argument(
        "--dst_path",
        type=str,
        default="./",
        help="destination path of generated dataset",
    )
    parser.add_argument(
        "--src_pascal_path",
        type=str,
        default="./path/to/source/pascal/VOCdevkit/VOC2012",
        help="source path of pascal dataset",
    )
    parser.add_argument(
        "--src_sbd_path",
        type=str,
        default="./path/to/source/sbd/benchmark_RELEASE/dataset",
        help="source path of sbd dataset",
    )
    args = parser.parse_args()

    # create folder
    dataset_path = Path(args.dst_path) / "PASCAL_SBD"
    os.makedirs(dataset_path, exist_ok=True)
    for folder in ["img", "gt", "list"]:
        os.makedirs(dataset_path / folder, exist_ok=True)

    # set original datasets path
    pascal_path = Path(args.src_pascal_path)
    sbd_path = Path(args.src_sbd_path)

    # get ids list
    pascal_train_ids = get_list_from_file(
        pascal_path / "ImageSets" / "Segmentation" / "train.txt"
    )
    pascal_val_ids = get_list_from_file(
        pascal_path / "ImageSets" / "Segmentation" / "val.txt"
    )
    pascal_all_ids = pascal_train_ids + pascal_val_ids
    sbd_train_ids = get_list_from_file(sbd_path / "train.txt")
    sbd_val_ids = get_list_from_file(sbd_path / "val.txt")
    sbd_all_ids = sbd_train_ids + sbd_val_ids
    pasbd_train_ids = list(
        (
            set(pascal_train_ids)
            | set(pascal_val_ids)
            | set(sbd_train_ids)
            | set(sbd_val_ids)
        )
        - set(pascal_val_ids)
    )
    pasbd_val_ids = list(pascal_val_ids)

    # process two splits
    process(pasbd_train_ids, "train")
    process(pasbd_val_ids, "val")
