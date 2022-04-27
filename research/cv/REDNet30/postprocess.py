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
"""postprocess"""
import os
import math
import glob
import argparse
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="/cache/data", help="dataset path.")
parser.add_argument("--save_path", type=str, default="/cache/output", help="output path.")
parser.add_argument("--bin_path", type=str, default="/cache/data", help="lr bin path.")
args = parser.parse_args()


def read_bin(bin_path):
    img = np.fromfile(bin_path, dtype=np.float32)
    num_pix = img.size
    img_shape = int(math.sqrt(num_pix / 3))
    if 1 * 3 * img_shape * img_shape != num_pix:
        raise RuntimeError(f'bin file error, it not output from dncnn network, {bin_path}')
    img = img.reshape(1, 3, img_shape, img_shape)
    return img


def read_bin_as_hwc(bin_path):
    nchw_img = read_bin(bin_path)
    chw_img = nchw_img[0]
    hwc_img = chw_img.transpose(1, 2, 0)
    return hwc_img


def PSNR(img1, img2):
    """metrics"""
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr


def SSIM(img1, img2):
    """metrics"""
    ssim = structural_similarity(img1, img2, data_range=255, multichannel=True)
    return ssim


def get_metric(ori_path, res_path):
    """metrics"""
    files = glob.glob(os.path.join(ori_path, "*"))
    names = []
    for i in files:
        names.append(i.split("/")[-1])

    # PSNR
    print("PSNR...")
    res = 0
    for i in names:
        ori = Image.open(os.path.join(ori_path, i))
        gen = Image.open(os.path.join(res_path, i))
        res += PSNR(np.array(ori), np.array(gen))
    psnr_res = res / len(names)

    # SSIM
    print("SSIM...")
    res = 0
    for i in names:
        ori = Image.open(os.path.join(ori_path, i))
        gen = Image.open(os.path.join(res_path, i))
        res += SSIM(np.array(ori), np.array(gen))
    ssim_res = res / len(names)

    print("PSNR: ", psnr_res)
    print("SSIM: ", ssim_res)


def run_post_process(dataset_path, save_path, bin_path):
    """run post process """
    files = os.listdir(dataset_path)
    files.sort()
    for file in files:

        file_name = file.split('.')[0]
        bin_file = os.path.join(bin_path, file_name + "_0.bin")
        sr = read_bin_as_hwc(bin_file)
        out_img = sr

        out_img = np.clip(out_img, 0, 255)
        out_img = np.uint8(out_img)
        out_img = Image.fromarray(out_img)
        out_img.save(os.path.join(save_path, file), quality=95)
    get_metric(dataset_path, save_path)


if __name__ == "__main__":
    run_post_process(args.dataset_path, args.save_path, args.bin_path)
