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
"""Preprocess for 310 inference: transform cityscapes to bin."""
import os
import argparse

from src.cityscapes import Cityscapes
from src.config import config_hrnetv2_w48 as config


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Cityscapes preprocess for OCRNet.")
    parser.add_argument("--data_path", type=str, help="Storage path of dataset.")
    parser.add_argument("--train_path", type=str, help="Storage path of bin files.")
    args = parser.parse_args()

    return args


def export_cityscapes_to_bin(args):
    """Convert data format from png to bin."""
    image_path = os.path.join(args.train_path, "image")
    label_path = os.path.join(args.train_path, "label")
    os.makedirs(image_path)
    os.makedirs(label_path)
    dataset = Cityscapes(args.data_path,
                         num_samples=None,
                         num_classes=config.dataset.num_classes,
                         multi_scale=False,
                         flip=False,
                         ignore_label=config.dataset.ignore_label,
                         base_size=config.eval.base_size,
                         crop_size=config.eval.image_size,
                         downsample_rate=1,
                         scale_factor=16,
                         mean=config.dataset.mean,
                         std=config.dataset.std,
                         is_train=False)
    for i, data in enumerate(dataset):
        image = data[0]
        label = data[1]
        file_name = "cityscapes_val_" + str(i) + ".bin"
        image_file_path = os.path.join(image_path, file_name)
        label_file_path = os.path.join(label_path, file_name)
        image.tofile(image_file_path)
        label.tofile(label_file_path)
    print("Export bin files finished!")


if __name__ == "__main__":
    args_opt = parse_args()
    export_cityscapes_to_bin(args=args_opt)
