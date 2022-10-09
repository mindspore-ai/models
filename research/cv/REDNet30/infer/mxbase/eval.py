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
"""sdk infer"""
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image

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
    for i in tqdm(names):
        ori = Image.open(os.path.join(ori_path, i))
        gen = Image.open(os.path.join(res_path, i))
        res += PSNR(np.array(ori), np.array(gen))
    psnr_res = res / len(names)

    # SSIM
    print("SSIM...")
    res = 0
    for i in tqdm(names):
        ori = Image.open(os.path.join(ori_path, i))
        gen = Image.open(os.path.join(res_path, i))
        res += SSIM(np.array(ori), np.array(gen))
    ssim_res = res / len(names)

    print("PSNR: ", psnr_res)
    print("SSIM: ", ssim_res)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../data/input_cut_train",
                        help="path of input images directory")
    parser.add_argument("--output_dir", type=str, default="../data/result",
                        help="path of output images directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    get_metric(args.input_dir, args.output_dir)
