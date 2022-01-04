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
"""LIP dataset lst maker."""
import os
import argparse


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="LIP Dataset lst Maker.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root.")
    return parser.parse_args()


def maker():
    """Create LIP train.lst and val.lst, and output files would be save in root path."""
    args = parse_args()
    train_id_file = os.path.join(args.root, "train_id.txt")
    val_id_file = os.path.join(args.root, "val_id.txt")
    if not os.path.exists(train_id_file):
        raise FileNotFoundError(f"`{train_id_file}` is not found.")
    if not os.path.exists(val_id_file):
        raise FileNotFoundError(f"`{val_id_file}` is not found.")
    # create train.lst
    print("Create train.lst...")
    train_lst_file = os.path.join(args.root, "train.lst")
    with open(train_id_file, "r") as fr:
        with open(train_lst_file, "w") as fw:
            ids = fr.readlines()
            for sid in ids:
                sid = sid.strip()
                image = "train_images" + "/" + sid + ".jpg"
                label = "train_segmentations" + "/" + sid + ".png"
                sample = image + " " + label + "\n"
                fw.write(sample)
    print("train.lst has been created successfully!")
    # create val.lst
    print("Create val.lst...")
    val_lst_file = os.path.join(args.root, "val.lst")
    with open(val_id_file, "r") as fr:
        with open(val_lst_file, "w") as fw:
            ids = fr.readlines()
            for sid in ids:
                sid = sid.strip()
                image = "val_images" + "/" + sid + ".jpg"
                label = "val_segmentations" + "/" + sid + ".png"
                sample = image + " " + label + "\n"
                fw.write(sample)
    print("val.lst has been created successfully!")
    print("Done.")


if __name__ == "__main__":
    maker()
    print("Done.")
