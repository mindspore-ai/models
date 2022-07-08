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
""" eval_sdk.py """
import os
import numpy as np
import skimage.metrics

def read_file_list(input_file):
    """
    :param infer file content:
        1.bin 0
        2.bin 2
        ...
    :return image path list, label list
    """
    noisy_image_file = []
    clear_image_file = []
    if not os.path.exists(input_file):
        print('input file does not exists.')
    with open(input_file, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(',')
            noisy_file_name = line[0]
            clear_file_name = line[1]
            noisy_image_file.append(noisy_file_name)
            clear_image_file.append(clear_file_name)
    return noisy_image_file, clear_image_file

images_txt_path = "../data/dncnn_infer_data/dncnn_bs_1_label.txt"

noisy_image_files, clear_image_files = read_file_list(images_txt_path)

mean_psnr = 0
mean_ssim = 0
count = 0

for index, out_file in enumerate(clear_image_files):
    out_path = '../data/dncnn_infer_data/dncnn_bs_1_clear_bin/' + out_file
    clear = np.fromfile(out_path, dtype=np.uint8).reshape(1, 256, 256)
    noisy = np.fromfile(out_path.replace('clear', 'noisy'), dtype=np.float32).reshape(1, 256, 256)
    # get denoised image
    residual = np.fromfile('./result/output_{}.bin'.format(index), dtype=np.float32).reshape(1, 256, 256)
    denoised = np.clip(noisy - residual, 0, 255).astype("uint8")
    denoised = np.squeeze(denoised)
    clear = np.squeeze(clear)
    noisy = np.squeeze(noisy)

    # calculate psnr
    mse = np.mean((clear - denoised) ** 2)
    psnr = 10 * np.log10(255 * 255 / mse)
    # calculate ssim
    ssim = skimage.metrics.structural_similarity(clear, denoised, data_range=255)  # skimage 0.18

    mean_psnr += psnr
    mean_ssim += ssim
    count += 1

mean_psnr = mean_psnr / count
mean_ssim = mean_ssim / count
print("mean psnr", mean_psnr)
print("mean_ssim", mean_ssim)
