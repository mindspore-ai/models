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
"""Pre-process for 310 inference: dataset preprocess."""
import os
import argparse

from src.dataset.dataset_generator import create_seg_dataset


def parse_args():
    """Preprocess parameters from command line."""
    parser = argparse.ArgumentParser(description="Cityscapes preprocess for HRNet-seg.")
    parser.add_argument("--data_path", type=str, help="Storage path of dataset.")
    parser.add_argument("--dataset", type=str, default="cityscapes")
    parser.add_argument("--train_path", type=str, help="Storage path of bin files.")
    args = parser.parse_args()

    return args


def export_cityscapes_to_bin(args):
    """Convert data format from png to bin."""
    image_path = os.path.join(args.train_path, "image")
    label_path = os.path.join(args.train_path, "label")
    os.makedirs(image_path)
    os.makedirs(label_path)
    loader, _, _, _ = create_seg_dataset(
        args.dataset, args.data_path, is_train=False, raw=True)
    for i, data in enumerate(loader):
        image = data[0]
        label = data[1]
        file_name = "cityscapes_val_" + str(i) + ".bin"
        image_file_path = os.path.join(image_path, file_name)
        label_file_path = os.path.join(label_path, file_name)
        image.tofile(image_file_path)
        label.tofile(label_file_path)
    print("Export bin files finished!", flush=True)


if __name__ == "__main__":
    args_opt = parse_args()
    export_cityscapes_to_bin(args=args_opt)
