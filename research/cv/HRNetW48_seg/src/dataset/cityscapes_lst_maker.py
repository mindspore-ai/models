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
"""Cityscapes dataset lst maker."""
import os
import argparse


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="Cityscapes Dataset lst Maker.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root.")
    return parser.parse_args()


def maker():
    """Create Cityscapes train.lst and val.lst, and output files would be save in root path."""
    args = parse_args()
    if not os.path.exists(os.path.join(args.root, "gtFine")):
        raise NotADirectoryError("`gtFine` is not in root path.")
    if not os.path.exists(os.path.join(args.root, "leftImg8bit")):
        raise NotADirectoryError("`leftImg8bit` is not in root path.")
    # create train.lst
    print("Create train.lst...")
    train_lst_file = os.path.join(args.root, "train.lst")
    prefix = "leftImg8bit/train/"
    with open(train_lst_file, "w") as fw:
        for city in os.listdir(os.path.join(args.root, prefix)):
            temp = prefix + city + "/"
            for image in os.listdir(os.path.join(args.root, temp)):
                img = temp + image
                msk = img.replace("_leftImg8bit", "_gtFine_labelIds").replace("leftImg8bit", "gtFine")
                sample = img + " " + msk + "\n"
                fw.write(sample)
    print("train.lst has been created successfully!")
    # create val.lst
    print("Create val.lst...")
    val_lst_file = os.path.join(args.root, "val.lst")
    prefix = "leftImg8bit/val/"
    with open(val_lst_file, "w") as fw:
        for city in os.listdir(os.path.join(args.root, prefix)):
            temp = prefix + city + "/"
            for image in os.listdir(os.path.join(args.root, temp)):
                img = temp + image
                msk = img.replace("_leftImg8bit", "_gtFine_labelIds").replace("leftImg8bit", "gtFine")
                sample = img + " " + msk + "\n"
                fw.write(sample)
    print("val.lst has been created successfully!")


if __name__ == "__main__":
    maker()
    print("Done.")
