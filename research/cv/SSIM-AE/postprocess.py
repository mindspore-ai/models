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

import os
import cv2
import numpy as np
from skimage import morphology
from skimage.metrics import structural_similarity as ssim

from src.utils import patch2img, read_img, get_depressing_mask, bg_mask, set_img_color
from src.eval_utils import apply_eval
from model_utils.config import config as cfg


def generate_single(test_img, decoded_img, img_name):
    """Generate visual images."""
    if test_img.shape[:2] == (cfg.crop_size, cfg.crop_size):
        decoded_img = np.transpose(decoded_img, (0, 2, 3, 1))
    else:
        decoded_img = np.transpose(decoded_img, (0, 2, 3, 1))
        decoded_img = patch2img(decoded_img, cfg.im_resize, cfg.crop_size, cfg.stride)

    rec_img = np.reshape((decoded_img * 255.0).astype(np.uint8), test_img.shape)

    if cfg.image_level:
        if cfg.grayscale:
            ssim_residual_map = 1 - ssim(test_img, rec_img, win_size=11, full=True)[1]
            l1_residual_map = np.abs(test_img / 255.0 - rec_img / 255.0)
        else:
            ssim_residual_map = ssim(test_img, rec_img, win_size=11, full=True, multichannel=True)[1]
            ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
            l1_residual_map = np.mean(np.abs(test_img / 255.0 - rec_img / 255.0), axis=2)
    else:
        if cfg.grayscale:
            ssim_residual_map = (
                1
                - ssim(
                    test_img,
                    rec_img,
                    win_size=11,
                    data_range=1,
                    gradient=False,
                    full=True,
                    gaussian_weights=True,
                    sigma=7,
                )[1]
            )
            l1_residual_map = np.abs(test_img / 255.0 - rec_img / 255.0)
        else:
            ssim_residual_map = ssim(
                test_img,
                rec_img,
                win_size=11,
                data_range=1,
                gradient=False,
                full=True,
                channel_axis=2,
                gaussian_weights=True,
                sigma=7,
            )[1]
            ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
            l1_residual_map = np.mean(np.abs(test_img / 255.0 - rec_img / 255.0), axis=2)
    ssim_residual_map *= cfg.depr_mask
    mask = np.zeros((cfg.mask_size, cfg.mask_size))
    if cfg.ssim_threshold < 0 or cfg.l1_threshold < 0:
        print(
            "[WARNING] Please check the ssim_threshold:{} and l1_threshold:{}".format(
                cfg.ssim_threshold, cfg.l1_threshold
            )
        )
    mask[ssim_residual_map > cfg.ssim_threshold] = 1
    mask[l1_residual_map > cfg.l1_threshold] = 1
    if cfg.bg_mask == "B":
        bg_m = bg_mask(test_img.copy(), 50, cv2.THRESH_BINARY, cfg.grayscale)
        mask *= bg_m
    elif cfg.bg_mask == "W":
        bg_m = bg_mask(test_img.copy(), 200, cv2.THRESH_BINARY_INV, cfg.grayscale)
        mask *= bg_m
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask = mask * 255
    vis = set_img_color(test_img.copy(), mask, weight_foreground=0.3, grayscale=cfg.grayscale)
    base_dir = os.path.dirname(os.path.join(cfg.save_dir, img_name))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_residual.png"), mask)
    cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_origin.png"), test_img)
    cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_rec.png"), rec_img)
    cv2.imwrite(os.path.join(cfg.save_dir, img_name + "_visual.png"), vis)


def generate_result():
    get_depressing_mask(cfg)
    channel = 1 if cfg.grayscale else 3
    batch_size = ((cfg.mask_size - cfg.crop_size) // cfg.stride + 1) ** 2
    img_shape = (batch_size, channel, cfg.crop_size, cfg.crop_size)
    with open("310_file_pair.txt", "r") as f:
        for line in f.readlines():
            img_name, _, bin_path = line.strip().split()
            test_img = read_img(os.path.join(cfg.test_dir, img_name + cfg.img_suffix), cfg.grayscale)
            if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
                test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
            if cfg.im_resize != cfg.mask_size:
                tmp = (cfg.im_resize - cfg.mask_size) // 2
                test_img = test_img[tmp : tmp + cfg.mask_size, tmp : tmp + cfg.mask_size]
            decoded_img = np.fromfile(bin_path, dtype=np.float32).reshape(img_shape)
            generate_single(test_img, decoded_img, img_name)


if __name__ == "__main__":
    generate_result()
    print("Generate results at", cfg.save_dir)
    apply_eval(cfg)
