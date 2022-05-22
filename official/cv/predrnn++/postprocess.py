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

'''
postprocess script.
'''

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import metrics
from config import config
from data_provider import preprocess
from skimage.metrics import structural_similarity

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--input0_path', type=str, default='')
    args_opt = parser.parse_args()

    file_num = len(os.listdir(args_opt.input0_path))

    batch_id = 0
    avg_mse = 0
    img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []

    for i in range(config.seq_length - config.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)

    for item in range(int(file_num)):

        batch_id = batch_id + 1

        f_name1 = "predrnn++_bs" + str(config.batch_size) + "_" + str(item) + ".bin"
        test_ims = np.fromfile(os.path.join(args_opt.input0_path, f_name1), np.float32).reshape(8, 20, 16, 16, 16)
        test_ims_tp = np.transpose(test_ims, (0, 1, 3, 4, 2))
        test_ims_ori = preprocess.reshape_patch_back(test_ims_tp, config.patch_size)

        f_name2 = "predrnn++_bs" + str(config.batch_size) + "_" + str(item) + "_" + "0" + ".bin"
        gen_images = np.fromfile(os.path.join(args_opt.result_dir, f_name2), np.float32).reshape(8, 19, 16, 16, 16)
        gen_images = np.transpose(gen_images, (0, 1, 3, 4, 2))
        img_gen = preprocess.reshape_patch_back(gen_images[:, 9:], config.patch_size)

        for i in range(config.seq_length - config.input_length):

            x = test_ims_ori[:, i + config.input_length, :, :, 0]
            gx = img_gen[:, i, :, :, 0]
            fmae[i] += metrics.batch_mae_frame_float(gx, x)

            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)

            for b in range(config.batch_size):
                sharp[i] += np.max(cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True)
                ssim[i] += score

    avg_mse = avg_mse / (batch_id*config.batch_size)
    ssim = np.asarray(ssim, dtype=np.float32)/(config.batch_size*batch_id)
    psnr = np.asarray(psnr, dtype=np.float32)/batch_id
    fmae = np.asarray(fmae, dtype=np.float32)/batch_id
    sharp = np.asarray(sharp, dtype=np.float32)/(config.batch_size*batch_id)

    print('mse per frame: ' + str(avg_mse/config.input_length))
    for i in range(config.seq_length - config.input_length):
        print(img_mse[i] / (batch_id*config.batch_size))

    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(config.seq_length - config.input_length):
        print(ssim[i])

    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(config.seq_length - config.input_length):
        print(psnr[i])

    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(config.seq_length - config.input_length):
        print(fmae[i])

    print('sharpness per frame: ' + str(np.mean(sharp)))
    for i in range(config.seq_length - config.input_length):
        print(sharp[i])
