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
"""preprocess"""
import os
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Preporcess")
parser.add_argument("--dataset_path", type=str, default="/cache/data", help="dataset path.")
parser.add_argument("--dataset_type", type=str, default="Set5", help="dataset type.")
parser.add_argument("--save_path", type=str, default="/cache/data", help="save lr dataset path.")
parser.add_argument("--scale", type=int, default="2", help="scale.")
args = parser.parse_args()

MAX_HR_SIZE = 2040


def padding(img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {img.shape}")
    if dh != 0 or dw != 0:
        img = np.pad(img, ((0, int(dh)), (0, int(dw)), (0, 0)), "constant")
    return img


def run_pre_process(dataset_path, dataset_type, scale, save_path):
    """run pre process"""
    lr_path = os.path.join(dataset_path, dataset_type, "LR_bicubic/X" + str(scale))
    files = os.listdir(lr_path)
    for file in files:
        lr = Image.open(os.path.join(lr_path, file))
        lr = lr.convert('RGB')
        lr = np.array(lr)
        target_shape = [MAX_HR_SIZE / scale, MAX_HR_SIZE / scale]
        img = padding(lr, target_shape)
        save_lr_path = os.path.join(save_path, file)
        os.makedirs(save_path, exist_ok=True)
        Image.fromarray(img).save(save_lr_path)


if __name__ == "__main__":
    run_pre_process(args.dataset_path, args.dataset_type, args.scale, args.save_path)
