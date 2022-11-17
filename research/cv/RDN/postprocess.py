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
"""postprocess"""
import os
import math
import argparse
import numpy as np
from src.metrics import calc_psnr
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/cache/data", help="dataset path.")
parser.add_argument("--dataset_type", type=str, default="Set5", help="dataset type.")
parser.add_argument("--bin_path", type=str, default="/cache/data", help="lr bin path.")
parser.add_argument("--scale", type=int, default="2", help="scale.")
args = parser.parse_args()


def unpadding(img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    if img_h > h:
        img = img[:h, :, :]
    if img_w > w:
        img = img[:, :w, :]
    return img


def read_bin(bin_path):
    img = np.fromfile(bin_path, dtype=np.float32)
    num_pix = img.size
    img_shape = int(math.sqrt(num_pix // 3))
    if 1 * 3 * img_shape * img_shape != num_pix:
        raise RuntimeError(f'bin file error, it not output from rcan network, {bin_path}')
    img = img.reshape(1, 3, img_shape, img_shape)
    return img


def read_bin_as_hwc(bin_path):
    nchw_img = read_bin(bin_path)
    chw_img = np.squeeze(nchw_img)
    hwc_img = chw_img.transpose(1, 2, 0)
    return hwc_img


def float_to_uint8(img):
    img = np.array(img)
    clip_img = np.clip(img, 0, 255)
    round_img = np.round(clip_img)
    uint8_img = round_img.astype(np.uint8)
    return uint8_img


def img_transpose(img):
    img = np.array([img.transpose(2, 0, 1)], np.float32)
    return img


def run_post_process(dataset_path, dataset_type, bin_path, scale):
    """run post process """
    files_path = os.path.join(dataset_path, dataset_type, "HR/")
    files = os.listdir(files_path)
    num = 0
    psnrs = 0
    avg_psnr = 0
    for file in files:
        file_name = file.split('.')[0]
        bin_file = os.path.join(bin_path, file_name + "x2_0.bin")
        sr = read_bin_as_hwc(bin_file)
        sr = float_to_uint8(sr)
        hr_path = os.path.join(files_path + file_name + '.png')
        hr = Image.open(hr_path)
        hr = hr.convert('RGB')
        hr = np.array(hr, dtype=np.float32)
        sr = unpadding(sr, hr.shape)
        sr = np.array(sr, dtype=np.float32)
        sr = img_transpose(sr)
        hr = img_transpose(hr)
        psnr = calc_psnr(sr, hr, scale, 255.0)
        psnrs += psnr
        num += 1
    avg_psnr = psnrs / num
    print("psnr: ", avg_psnr)
    print("post_process success", flush=True)


if __name__ == "__main__":
    run_post_process(args.dataset_path, args.dataset_type, args.bin_path, args.scale)
